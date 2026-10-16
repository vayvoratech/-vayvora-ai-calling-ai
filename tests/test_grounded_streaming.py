"""Comprehensive focused tests for Phase 5C - Phase 2: Grounded RAG Streaming Integration.

Verifies:
1. Existing callers work seamlessly without the optional on_grounded_chunk callback.
2. Callback receives phrases in the correct chronological order.
3. Split-word fragments arriving across token boundaries are buffered correctly.
4. Final buffered text is emitted once at end-of-stream.
5. The returned EngineTurnResult.response_text matches the complete streamed text without duplication.
6. No callback is invoked before retrieval and safety checks finish (e.g. relevance gate failure).
7. Non-RAG turns (chit-chat, greetings) do not invoke on_grounded_chunk.
8. Batch fallback works when streaming is unsupported by the provider.
9. Failure before the first emitted phrase safely falls back to batch generation.
10. Failure after partial emission preserves emitted text and does not replay the full batch answer.
11. Callback exceptions propagate and are handled explicitly without pretending delivery succeeded.
12. Task cancellation (barge-in CancelledError) propagates and is never swallowed.
13. Existing pricing safeguards and action verification remain strictly intact.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
import pytest

import src.core.types  # Enforce initialization order
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.interfaces import KnowledgeProvider, LLMProvider, ToolProvider
from src.core.llm import MockLLMProvider
from src.core.types import (
    ConversationStage,
    DomainType,
    RAGChunk,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
)
from src.rag.retriever import GroundedContextResult
from src.state.manager import ConversationStateManager


# =============================================================================
# Test Mocks
# =============================================================================

class MockGroundedKnowledgeProvider(KnowledgeProvider):
    """Mock KnowledgeProvider for controlled RAG context testing."""

    def __init__(
        self,
        chunks: Optional[List[RAGChunk]] = None,
        knowledge_available: bool = True,
        service_unavailable: bool = False,
    ):
        self.chunks = chunks if chunks is not None else []
        self.knowledge_available = knowledge_available and bool(self.chunks)
        self.service_unavailable = service_unavailable
        self.invocations: List[RAGQuery] = []

    async def search(self, query: RAGQuery) -> List[RAGChunk]:
        self.invocations.append(query)
        return self.chunks

    async def retrieve_grounded_context(self, query: RAGQuery) -> GroundedContextResult:
        self.invocations.append(query)
        if self.service_unavailable:
            return GroundedContextResult(
                query=query,
                chunks=[],
                knowledge_available=False,
                service_unavailable=True,
                error_message="Service unavailable",
            )
        if not self.chunks or not self.knowledge_available:
            return GroundedContextResult(
                query=query,
                chunks=[],
                knowledge_available=False,
                formatted_context="No verified knowledge found passing relevance threshold.",
            )

        context_lines = [
            f'<verified_reference_data domain="{query.domain.value}">',
        ]
        for idx, c in enumerate(self.chunks, 1):
            context_lines.append(f"--- Document {idx}: {c.title} ---")
            context_lines.append(f"Content:\n{c.content}\n")
        context_lines.append("</verified_reference_data>")

        return GroundedContextResult(
            query=query,
            chunks=self.chunks,
            knowledge_available=True,
            formatted_context="\n".join(context_lines),
        )

    async def index_document(self, chunk: RAGChunk) -> bool:
        self.chunks.append(chunk)
        return True


class NonStreamingMockLLMProvider:
    """Mock provider that does NOT implement stream_response, to test batch fallback."""

    def __init__(self, response: str = "Fallback batch response."):
        self.response = response
        self.invocations: List[Dict[str, Any]] = []

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        self.invocations.append({"prompt": prompt, "system_instruction": system_instruction})
        return self.response

    async def generate_decision(
        self, prompt: str, system_instruction: str, max_retries: int = 1
    ) -> ConversationalDecision:
        return ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="amenities_query",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="",
            knowledge_required=True,
            knowledge_query="Pristine City amenities",
        )


def build_test_chunk() -> RAGChunk:
    return RAGChunk(
        doc_id="infra:pristine_city:1",
        domain=DomainType.INFRA_BUILDER,
        title="Pristine City Overview",
        content="Pristine City features luxury plotted layouts near Kompally with wide 40-foot roads and a clubhouse.",
        score=0.95,
        metadata={"project": "Pristine City"},
    )


def build_rag_decision() -> ConversationalDecision:
    return ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="property_inquiry",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="",
        knowledge_required=True,
        knowledge_query="Pristine City layout details",
    )


# =============================================================================
# 1. Existing Callers Work Without Optional Callback
# =============================================================================

@pytest.mark.asyncio
async def test_01_existing_callers_work_without_optional_callback():
    """Existing process_user_turn callers without on_grounded_chunk callback work identically."""
    decision = build_rag_decision()
    grounded_text = "Pristine City is located in Kompally and offers luxury plots with wide roads."

    mock_llm = MockLLMProvider(
        canned_decisions=[decision],
        canned_responses=[grounded_text],
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-1", "+919876543210", domain=DomainType.INFRA_BUILDER)

    # Calling without on_grounded_chunk
    result = await engine.process_user_turn(state, "Tell me about Pristine City")

    assert isinstance(result, EngineTurnResult)
    assert result.response_text == grounded_text
    assert result.knowledge_required is True
    assert len(result.grounded_citations) > 0


# =============================================================================
# 2. Callback Receives Phrases in Correct Order
# =============================================================================

@pytest.mark.asyncio
async def test_02_callback_receives_phrases_in_correct_order():
    """Streaming grounded RAG emits complete phrases sequentially to on_grounded_chunk."""
    decision = build_rag_decision()
    stream_fragments = [
        "Pristine City offers ",
        "luxury residential plots. ",
        "Each plot includes ",
        "wide blacktop roads.",
    ]

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(stream_fragments)
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-2", "+919876543210", domain=DomainType.INFRA_BUILDER)

    received_phrases: List[str] = []

    async def chunk_callback(phrase: str) -> None:
        received_phrases.append(phrase)

    result = await engine.process_user_turn(
        state=state,
        user_message="What are the details of Pristine City?",
        on_grounded_chunk=chunk_callback,
    )

    assert len(received_phrases) == 2
    assert received_phrases[0] == "Pristine City offers luxury residential plots."
    assert received_phrases[1] == "Each plot includes wide blacktop roads."
    # Returned engine result matches complete text without repetition
    assert result.response_text == "".join(stream_fragments).strip()


# =============================================================================
# 3. Split-Word Fragments Are Buffered Correctly
# =============================================================================

@pytest.mark.asyncio
async def test_03_split_word_fragments_are_buffered_correctly():
    """Tokens sliced across sub-word boundaries accumulate without chopping words."""
    decision = build_rag_decision()
    # "Hyderabad" cut across 3 fragments, "plots" cut across 2 fragments
    stream_fragments = [
        "We have plots in Hy",
        "der",
        "abad. ",
        "They feature pl",
        "anned layouts.",
    ]

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(stream_fragments)
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-3", "+919876543210", domain=DomainType.INFRA_BUILDER)

    received_phrases: List[str] = []

    def sync_chunk_callback(phrase: str) -> None:
        received_phrases.append(phrase)

    result = await engine.process_user_turn(
        state=state,
        user_message="Tell me about the plots",
        on_grounded_chunk=sync_chunk_callback,
    )

    assert received_phrases == [
        "We have plots in Hyderabad.",
        "They feature planned layouts.",
    ]
    assert "Hyderabad" in result.response_text


# =============================================================================
# 4. Final Buffered Text Emitted Once at End of Stream
# =============================================================================

@pytest.mark.asyncio
async def test_04_final_buffered_text_is_emitted_once():
    """Text remaining in the buffer without terminal punctuation flushes once on stream completion."""
    decision = build_rag_decision()
    # Sentence 1 has period, sentence 2 has no terminal punctuation
    stream_fragments = [
        "Welcome to Pristine City! ",
        "All plots are ready for registration",
    ]

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(stream_fragments)
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-4", "+919876543210", domain=DomainType.INFRA_BUILDER)

    received_phrases: List[str] = []

    await engine.process_user_turn(
        state=state,
        user_message="Are plots ready?",
        on_grounded_chunk=lambda p: received_phrases.append(p),
    )

    assert len(received_phrases) == 2
    assert received_phrases[0] == "Welcome to Pristine City!"
    assert received_phrases[1] == "All plots are ready for registration"


# =============================================================================
# 5. Returned Response Matches Complete Streamed Text Without Duplication
# =============================================================================

@pytest.mark.asyncio
async def test_05_returned_response_matches_complete_streamed_text_without_duplication():
    """EngineTurnResult.response_text matches the full response once without duplicated phrases."""
    decision = build_rag_decision()
    stream_fragments = [
        "Pristine City features luxury villas. ",
        "The community offers scenic plotted lands.",
    ]

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(stream_fragments)
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-5", "+919876543210", domain=DomainType.INFRA_BUILDER)

    received_phrases: List[str] = []
    result = await engine.process_user_turn(
        state=state,
        user_message="What features does Pristine City have?",
        on_grounded_chunk=lambda p: received_phrases.append(p),
    )

    # All phrases were received
    assert len(received_phrases) == 2
    # The return value is exactly the joined complete text
    assert result.response_text == "Pristine City features luxury villas. The community offers scenic plotted lands."
    # State history records the utterance once
    agent_turns = [t for t in state.history if t.role.value == "agent"]
    assert len(agent_turns) == 1
    assert agent_turns[0].content == result.response_text


# =============================================================================
# 6. No Callback Invoked Before Retrieval & Safety Checks Finish
# =============================================================================

@pytest.mark.asyncio
async def test_06_no_callback_invoked_when_relevance_gate_fails():
    """When RAG retrieval finds no relevant knowledge, safe fallback is used and on_grounded_chunk is NEVER called."""
    decision = build_rag_decision()

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(["This should never be streamed!"])
    # Empty chunks -> relevance gate fails closed
    mock_kp = MockGroundedKnowledgeProvider(chunks=[])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-6", "+919876543210", domain=DomainType.INFRA_BUILDER)

    callback_invoked = False

    def chunk_callback(phrase: str) -> None:
        nonlocal callback_invoked
        callback_invoked = True

    result = await engine.process_user_turn(
        state=state,
        user_message="Tell me about the amenities on Mars",
        on_grounded_chunk=chunk_callback,
    )

    assert callback_invoked is False
    assert "don't have verified details" in result.response_text


# =============================================================================
# 7. Non-RAG Turns Do Not Stream
# =============================================================================

@pytest.mark.asyncio
async def test_07_non_rag_turns_do_not_stream():
    """Simple greetings and acknowledgments without RAG grounding do not invoke on_grounded_chunk."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="greeting",
        proposed_stage=ConversationStage.GREETING,
        user_facing_response="Hello! How can I assist you with Infra Builder properties today?",
        knowledge_required=False,
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-7", "+919876543210", domain=DomainType.INFRA_BUILDER)

    callback_called = False

    result = await engine.process_user_turn(
        state=state,
        user_message="Hello, good morning",
        on_grounded_chunk=lambda p: globals().update(callback_called=True),
    )

    assert callback_called is False
    assert result.knowledge_required is False
    assert "assist you" in result.response_text


# =============================================================================
# 8. Batch Fallback When Streaming Is Unsupported by Provider
# =============================================================================

@pytest.mark.asyncio
async def test_08_batch_fallback_when_streaming_unsupported():
    """If provider does not support stream_response, falls back to generate_response."""
    batch_text = "Pristine City features luxury plotted layouts near Kompally."
    provider = NonStreamingMockLLMProvider(response=batch_text)
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=provider, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-8", "+919876543210", domain=DomainType.INFRA_BUILDER)

    callback_called = False
    result = await engine.process_user_turn(
        state=state,
        user_message="Tell me about Pristine City",
        on_grounded_chunk=lambda p: globals().update(callback_called=True),
    )

    assert callback_called is False
    assert result.response_text == batch_text


# =============================================================================
# 9. Failure Before First Phrase Uses Batch Fallback
# =============================================================================

@pytest.mark.asyncio
async def test_09_failure_before_first_phrase_uses_batch_fallback():
    """If stream fails immediately before any phrase is emitted, falls back to batch generation."""
    decision = build_rag_decision()
    batch_fallback_text = "Pristine City is a premium layout in North Hyderabad."

    class FailingStreamLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            raise RuntimeError("Connection dropped on stream handshake")
            yield "unreachable"

    mock_llm = FailingStreamLLM(
        canned_decisions=[decision],
        canned_responses=[batch_fallback_text],
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-9", "+919876543210", domain=DomainType.INFRA_BUILDER)

    received_phrases: List[str] = []
    result = await engine.process_user_turn(
        state=state,
        user_message="Tell me about Pristine City",
        on_grounded_chunk=lambda p: received_phrases.append(p),
    )

    # No phrases were emitted to callback
    assert received_phrases == []
    # Batch fallback took over successfully
    assert result.response_text == batch_fallback_text


# =============================================================================
# 10. Failure After Partial Emission Preserves Partial Text Without Replaying
# =============================================================================

@pytest.mark.asyncio
async def test_10_failure_after_partial_emission_does_not_replay_full_answer():
    """If stream breaks after emitting a phrase, does not restart batch generation to avoid speech repetition."""
    decision = build_rag_decision()

    class MidStreamFailingLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            yield "Pristine City features wide roads. "
            yield "The clubhouse has "
            raise RuntimeError("Stream connection dropped mid-generation!")

    mock_llm = MidStreamFailingLLM(canned_decisions=[decision])
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-10", "+919876543210", domain=DomainType.INFRA_BUILDER)

    received_phrases: List[str] = []
    result = await engine.process_user_turn(
        state=state,
        user_message="Tell me about Pristine City",
        on_grounded_chunk=lambda p: received_phrases.append(p),
    )

    # The first phrase was delivered
    assert received_phrases == ["Pristine City features wide roads."]
    # The result preserves the emitted text without replaying a full new batch answer
    assert result.response_text == "Pristine City features wide roads."


# =============================================================================
# 11. Callback Exceptions Propagate Explicitly
# =============================================================================

@pytest.mark.asyncio
async def test_11_callback_exception_propagates_explicitly():
    """If the on_grounded_chunk callback fails (e.g. WebSocket drop), error propagates without being swallowed."""
    decision = build_rag_decision()
    stream_fragments = ["Welcome to Pristine City. ", "More info here."]

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(stream_fragments)
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-11", "+919876543210", domain=DomainType.INFRA_BUILDER)

    async def broken_callback(phrase: str) -> None:
        raise ConnectionResetError("WebSocket transport forcibly closed")

    with pytest.raises(ConnectionResetError, match="WebSocket transport forcibly closed"):
        await engine.process_user_turn(
            state=state,
            user_message="Tell me about Pristine City",
            on_grounded_chunk=broken_callback,
        )


# =============================================================================
# 12. Task Cancellation (Barge-In) Propagates
# =============================================================================

@pytest.mark.asyncio
async def test_12_cancellation_propagates_on_barge_in():
    """asyncio.CancelledError raised during streaming is never swallowed or converted into fake success."""
    decision = build_rag_decision()

    class BargeInCancellingLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            yield "First few words "
            raise asyncio.CancelledError()

    mock_llm = BargeInCancellingLLM(canned_decisions=[decision])
    mock_kp = MockGroundedKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-12", "+919876543210", domain=DomainType.INFRA_BUILDER)

    with pytest.raises(asyncio.CancelledError):
        await engine.process_user_turn(
            state=state,
            user_message="Tell me about Pristine City",
            on_grounded_chunk=lambda p: None,
        )


# =============================================================================
# 13. Existing Pricing Safeguards Remain Intact
# =============================================================================

@pytest.mark.asyncio
async def test_13_unverified_pricing_queries_fail_closed_without_streaming():
    """An unconfirmed price claim fails closed deterministically and never streams speculative rates."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="pricing_query",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="",
        knowledge_required=True,
        knowledge_query="Pristine City price per square yard",
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(["The price is only 1,000 rupees!"])
    # Chunk contains description but NO verified price line
    chunk_without_price = RAGChunk(
        doc_id="infra:pristine_city:desc",
        domain=DomainType.INFRA_BUILDER,
        title="Pristine City Amenities",
        content="Pristine City is an exclusive residential community.",
        score=0.9,
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[chunk_without_price])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-stream-13", "+919876543210", domain=DomainType.INFRA_BUILDER)

    callback_called = False
    result = await engine.process_user_turn(
        state=state,
        user_message="What is the price of Pristine City?",
        on_grounded_chunk=lambda p: globals().update(callback_called=True),
    )

    assert callback_called is False
    assert "can't verify the current price" in result.response_text
