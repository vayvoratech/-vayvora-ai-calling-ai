"""Focused regression tests verifying that ungrounded first-pass responses cannot bypass RAG synthesis.

Covers:
1. A factual query marked knowledge_required=True cannot reuse an unsupported initial draft.
2. High keyword overlap between an initial draft and retrieved text cannot bypass grounded synthesis.
3. Relevant retrieved knowledge reaches the grounded synthesis path.
4. Insufficient or irrelevant retrieval does not cause unsupported facts to be returned.
5. A greeting or other non-knowledge response retains its existing behavior.
6. Existing price-query safeguards remain intact.
7. Existing tool/action flows are unaffected.
"""

from typing import Any, Dict, List, Optional
import pytest

import src.core.types  # Enforce initialization order
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult, is_reusable_first_response
from src.core.interfaces import KnowledgeProvider, ToolProvider
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


class MockGroundedKnowledgeProvider(KnowledgeProvider):
    """Configurable mock KnowledgeProvider for testing engine response selection."""

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
            f'<verified_reference_data domain="{query.domain.value}" untrusted="true">',
        ]
        for idx, c in enumerate(self.chunks, 1):
            context_lines.append(f"--- Document {idx}: {c.title} ---")
            context_lines.append(f"Source: {c.metadata.get('source_file', 'unknown')}")
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


class MockToolProvider(ToolProvider):
    """Mock tool provider for action verification tests."""

    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.executed_tools: List[ToolCallRequest] = []

    async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
        self.executed_tools.append(request)
        return ToolExecutionResult(
            tool_name=request.tool_name,
            success=self.should_succeed,
            data={"status": "delivered" if self.should_succeed else "failed"},
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "send_message", "description": "Send WhatsApp/SMS message"},
            {"name": "find_available_slots", "description": "Find calendar slots"},
            {"name": "create_calendar_event", "description": "Book a site visit"},
        ]


# =============================================================================
# 1. FACTUAL QUERY CANNOT REUSE UNSUPPORTED INITIAL DRAFT
# =============================================================================
@pytest.mark.asyncio
async def test_factual_query_cannot_reuse_unsupported_initial_draft():
    """Requirement 1: A factual query marked knowledge_required=True cannot reuse an unsupported initial draft."""
    unsupported_draft = "JB Pride features a full Olympic swimming pool, 4 tennis courts, and a cricket stadium."
    grounded_synthesis = "JB Pride offers a modern clubhouse and landscaped open gardens. How can I help you further?"

    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="amenities_query",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response=unsupported_draft,
        knowledge_required=True,
        knowledge_query="JB Pride sports amenities",
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision], canned_responses=[grounded_synthesis])
    verified_chunk = RAGChunk(
        doc_id="infra_builder:projects:jb_pride:1",
        domain=DomainType.INFRA_BUILDER,
        title="JB Infra JB Pride",
        content="JB Infra JB Pride provides a modern clubhouse, children's play area, and landscaped open gardens.",
        score=0.92,
        metadata={"source_file": "infra_builder/projects/jb_pride.md"},
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[verified_chunk])

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-grounding-1", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "What sports amenities does JB Pride have?")

    # Must NOT reuse the unsupported initial draft
    assert result.response_text != unsupported_draft
    assert "tennis courts" not in result.response_text.lower()
    assert "cricket stadium" not in result.response_text.lower()
    # Must use the grounded synthesis
    assert "modern clubhouse" in result.response_text.lower()
    # Second LLM call for grounded synthesis must have been invoked
    synthesis_invocations = [
        inv for inv in mock_llm.invocations
        if "prompt" in inv and "rag/grounded_answer.txt" not in inv and "<verified_reference_data" in inv["prompt"]
    ]
    assert len(synthesis_invocations) == 1


# =============================================================================
# 2. HIGH KEYWORD OVERLAP CANNOT BYPASS GROUNDED SYNTHESIS
# =============================================================================
@pytest.mark.asyncio
async def test_high_keyword_overlap_cannot_bypass_grounded_synthesis():
    """Requirement 2: High keyword overlap between an initial draft and retrieved text cannot bypass grounded synthesis."""
    # Draft contains words matching the chunk ("Pristine City", "gated community", "layout", "blacktop roads")
    # but invents "indoor badminton court"
    overlapping_draft = (
        "Pristine City is a gated community layout with blacktop roads and an indoor badminton court."
    )
    grounded_synthesis = (
        "Pristine City is a premium gated community layout featuring blacktop roads and underground drainage."
    )

    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="project_information",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response=overlapping_draft,
        knowledge_required=True,
        knowledge_query="Pristine City facilities",
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision], canned_responses=[grounded_synthesis])
    verified_chunk = RAGChunk(
        doc_id="infra_builder:projects:pristine_city:1",
        domain=DomainType.INFRA_BUILDER,
        title="JB Infra Pristine City",
        content=(
            "JB Infra Pristine City is a gated community layout near TSPA with blacktop roads, "
            "underground drainage, and street lighting."
        ),
        score=0.95,
        metadata={"source_file": "infra_builder/projects/pristine_city.md"},
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[verified_chunk])

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-grounding-2", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "What facilities are available at Pristine City?")

    # Even with high keyword overlap, the ungrounded draft must NOT be reused
    assert result.response_text != overlapping_draft
    assert "indoor badminton court" not in result.response_text.lower()
    assert result.response_text == grounded_synthesis


# =============================================================================
# 3. RELEVANT KNOWLEDGE REACHES GROUNDED SYNTHESIS PATH
# =============================================================================
@pytest.mark.asyncio
async def test_relevant_retrieved_knowledge_reaches_grounded_synthesis():
    """Requirement 3: Relevant retrieved knowledge reaches the grounded synthesis prompt and system instruction."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="location_query",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Draft location response",
        knowledge_required=True,
        knowledge_query="Pristine City location",
    )

    grounded_synthesis = "Pristine City is located near TSPA Appa Junction on the Vikarabad highway."
    mock_llm = MockLLMProvider(canned_decisions=[decision], canned_responses=[grounded_synthesis])

    unique_location_evidence = "near TSPA Appa Junction on the Vikarabad highway, Hyderabad"
    verified_chunk = RAGChunk(
        doc_id="infra_builder:projects:pristine_city:2",
        domain=DomainType.INFRA_BUILDER,
        title="Pristine City Location",
        content=f"Pristine City is strategically situated {unique_location_evidence}.",
        score=0.88,
        metadata={"source_file": "infra_builder/projects/pristine_city.md"},
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[verified_chunk])

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-grounding-3", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Where is Pristine City located?")

    # Verify that the second LLM call received the verified context
    synthesis_calls = [
        inv for inv in mock_llm.invocations
        if inv.get("system_instruction") and "GROUNDING RULES:" in inv["system_instruction"]
    ]
    assert len(synthesis_calls) == 1
    call = synthesis_calls[0]
    assert unique_location_evidence in call["prompt"]
    assert "<verified_reference_data" in call["prompt"]
    assert result.response_text == grounded_synthesis


# =============================================================================
# 4. INSUFFICIENT OR IRRELEVANT RETRIEVAL DOES NOT CAUSE UNSUPPORTED FACTS
# =============================================================================
@pytest.mark.asyncio
async def test_insufficient_retrieval_does_not_return_unsupported_facts():
    """Requirement 4: Insufficient or irrelevant retrieval returns safe fallback, not unsupported facts."""
    unsupported_hallucination = "Nature Valley has a private helipad and luxury clubhouse for residents."

    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="amenities_query",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response=unsupported_hallucination,
        knowledge_required=True,
        knowledge_query="Nature Valley helipad",
    )

    # Mock provider returns no chunks (relevance threshold failed)
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_kp = MockGroundedKnowledgeProvider(chunks=[], knowledge_available=False)

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-grounding-4", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Does Nature Valley have a helipad?")

    # Must NOT return the unsupported hallucination
    assert result.response_text != unsupported_hallucination
    assert "helipad" not in result.response_text.lower()
    # Must return the safe fallback
    assert "verified details" in result.response_text.lower() or "follow-up" in result.response_text.lower()


# =============================================================================
# 5. GREETINGS & NON-KNOWLEDGE RESPONSES RETAIN EXISTING BEHAVIOR
# =============================================================================
@pytest.mark.asyncio
async def test_greeting_and_non_knowledge_retains_existing_behavior():
    """Requirement 5: A greeting or other non-knowledge response retains its existing behavior."""
    greeting_text = "Good morning! How can I assist you with your property inquiry today?"

    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="greeting",
        proposed_stage=ConversationStage.GREETING,
        user_facing_response=greeting_text,
        knowledge_required=False,
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_kp = MockGroundedKnowledgeProvider(chunks=[])

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-grounding-5", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Good morning!")

    # Uses the greeting response directly
    assert result.response_text == greeting_text
    # No RAG search executed
    assert len(mock_kp.invocations) == 0
    # No second LLM call
    assert len(mock_llm.invocations) == 1


# =============================================================================
# 6. EXISTING PRICE-QUERY SAFEGUARDS REMAIN INTACT
# =============================================================================
@pytest.mark.asyncio
async def test_price_query_verified_line_quoted():
    """Requirement 6A: Pricing query with verified evidence line quotes exact line safely."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="price_query",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Draft price text",
        knowledge_required=True,
        knowledge_query="Nature Valley plot price",
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    price_chunk = RAGChunk(
        doc_id="infra_builder:projects:nature_valley:3",
        domain=DomainType.INFRA_BUILDER,
        title="Nature Valley Pricing",
        content="Plot Price: Rs. 12,500 per sq. yard onwards for standard plots.",
        score=0.90,
        metadata={"source_file": "infra_builder/projects/nature_valley.md"},
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[price_chunk])

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-grounding-6a", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "What is the price of Nature Valley plots?")

    # Strict pricing guard extracts the exact verified line
    assert "12,500" in result.response_text
    assert "sq. yard" in result.response_text.lower()


@pytest.mark.asyncio
async def test_price_query_unverified_fails_closed():
    """Requirement 6B: Pricing query without verified evidence line fails closed with refusal."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="price_query",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Pristine City plots cost 50 lakhs.",  # Hallucinated draft
        knowledge_required=True,
        knowledge_query="Pristine City price",
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    # Chunk has project info but NO verified price rate line
    non_price_chunk = RAGChunk(
        doc_id="infra_builder:projects:pristine_city:4",
        domain=DomainType.INFRA_BUILDER,
        title="Pristine City Overview",
        content="Pristine City is a 25-acre gated layout near TSPA Junction.",
        score=0.89,
        metadata={"source_file": "infra_builder/projects/pristine_city.md"},
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[non_price_chunk])

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-grounding-6b", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "What is the price of Pristine City plots?")

    # Must fail closed with verified refusal and follow-up offer
    assert "can't verify" in result.response_text.lower() or "unconfirmed rate" in result.response_text.lower()
    assert "50 lakhs" not in result.response_text.lower()


# =============================================================================
# 7. EXISTING TOOL / ACTION FLOWS UNAFFECTED
# =============================================================================
@pytest.mark.asyncio
async def test_existing_tool_action_flow_unaffected():
    """Requirement 7: Existing tool/action flows remain unaffected."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="book_site_visit",
        extracted_slots={"preferred_project": "JB Pride", "preferred_date": "tomorrow"},
        proposed_stage=ConversationStage.ACTION_CONFIRMATION,
        user_facing_response="Let me check available slots for JB Pride.",
        action_proposed=True,
        proposed_action=ProposedAction(
            tool_name="find_available_slots",
            arguments={"property_name": "JB Pride", "preferred_date": "tomorrow"},
            rationale="User requested site visit availability",
        ),
        knowledge_required=False,
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_tp = MockToolProvider(should_succeed=True)

    engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-grounding-7", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Can you check site visit availability for JB Pride tomorrow?")

    # Tool is executed and verified
    assert len(mock_tp.executed_tools) == 1
    assert mock_tp.executed_tools[0].tool_name == "find_available_slots"
    assert result.tool_result is not None
    assert result.tool_result.success is True


# =============================================================================
# 8. PHASE 5B-2: RAG DRAFT RESPONSE SUPPRESSION TESTS
# =============================================================================

def test_knowledge_required_decision_accepts_empty_user_facing_response():
    """Phase 5B-2: ConversationalDecision allows empty user_facing_response when knowledge_required=True."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="project_inquiry",
        knowledge_required=True,
        knowledge_query="Tell me about JB Pride",
        user_facing_response="",
    )
    assert decision.user_facing_response == ""
    assert decision.knowledge_required is True


def test_knowledge_required_decision_accepts_null_user_facing_response():
    """Phase 5B-2: ConversationalDecision normalizes None to empty string when knowledge_required=True."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="project_inquiry",
        knowledge_required=True,
        knowledge_query="Tell me about JB Pride",
        user_facing_response=None,
    )
    assert decision.user_facing_response == ""
    assert decision.knowledge_required is True


def test_non_rag_decision_rejects_empty_user_facing_response():
    """Phase 5B-2: Non-RAG decision raises validation error if user_facing_response is empty and not terminating."""
    import pydantic
    with pytest.raises((pydantic.ValidationError, ValueError)):
        ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="greeting",
            knowledge_required=False,
            user_facing_response="",
            suggested_termination=False,
        )

    # However, suggested_termination=True permits empty or minimal response
    term_decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="hangup",
        knowledge_required=False,
        user_facing_response="",
        suggested_termination=True,
    )
    assert term_decision.suggested_termination is True


@pytest.mark.asyncio
async def test_empty_rag_draft_still_runs_grounded_synthesis():
    """Phase 5B-2: Suppressing draft response on RAG turns still runs full grounded RAG synthesis."""
    chunk = RAGChunk(
        doc_id="pride-loc-1",
        domain=DomainType.INFRA_BUILDER,
        score=0.95,
        content="JB Pride is a luxury gated community located in Kompally with 150 ready-to-move villas.",
        metadata={"project": "JB Pride", "source_file": "infra_builder/projects/jb_pride.md"},
        title="JB Pride Location",
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[chunk])

    # Initial decision has EMPTY draft response
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="project_inquiry",
        extracted_slots={"preferred_project": "JB Pride"},
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="",
        knowledge_required=True,
        knowledge_query="JB Pride Kompally",
    )
    grounded_synthesis_output = (
        "JB Pride is located in Kompally and features 150 luxury villas. "
        "Would you like more details on villa sizes or pricing?"
    )

    mock_llm = MockLLMProvider(
        canned_decisions=[decision],
        canned_responses=[grounded_synthesis_output],
    )
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-rag-empty-draft-1", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Where is JB Pride located?")

    assert result.response_text == grounded_synthesis_output
    assert "Kompally" in result.response_text
    assert len(mock_kp.invocations) == 1
    assert result.latency_breakdown is not None
    assert "grounded_llm_ms" in result.latency_breakdown


@pytest.mark.asyncio
async def test_empty_rag_draft_insufficient_evidence_uses_safe_fallback():
    """Phase 5B-2: When draft is suppressed and RAG yields no relevant chunks, engine uses safe fallback."""
    mock_kp = MockGroundedKnowledgeProvider(chunks=[], knowledge_available=False)

    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="project_inquiry",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="",
        knowledge_required=True,
        knowledge_query="Unknown Project details",
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-rag-empty-draft-2", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Tell me about Unknown Project")

    # Safe fallback response must be returned, never empty
    assert result.response_text != ""
    assert "checked our knowledge base" in result.response_text.lower() or "don't have verified details" in result.response_text.lower()


@pytest.mark.asyncio
async def test_empty_rag_draft_unverified_price_fails_closed():
    """Phase 5B-2: Pricing query with empty draft and unverified price evidence fails closed."""
    non_price_chunk = RAGChunk(
        doc_id="pristine-desc-1",
        domain=DomainType.INFRA_BUILDER,
        title="Pristine City Overview",
        content="Pristine City is a 25-acre gated layout near TSPA Junction.",
        score=0.89,
        metadata={"source_file": "infra_builder/projects/pristine_city.md"},
    )
    mock_kp = MockGroundedKnowledgeProvider(chunks=[non_price_chunk])

    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="price_query",
        extracted_slots={"preferred_project": "Pristine City"},
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="",
        knowledge_required=True,
        knowledge_query="Pristine City price",
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-rag-empty-draft-3", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "What is the price of Pristine City plots?")

    assert result.response_text != ""
    assert "can't verify" in result.response_text.lower() or "unconfirmed rate" in result.response_text.lower()


@pytest.mark.asyncio
async def test_unknown_domain_or_simple_ack_with_empty_draft_provides_safe_fallback():
    """Phase 5B-2: If RAG was cancelled because of simple ack or unknown domain, safe fallback is provided."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="acknowledgment",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="",
        knowledge_required=True,  # Mistakenly set by LLM on simple ack
        knowledge_query=None,
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    engine = ConversationEngine(llm_provider=mock_llm)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-rag-empty-draft-4", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "okay")

    # Engine sets knowledge_required=False because 'okay' is a simple ack, and supplies a safe fallback
    assert result.response_text != ""
    assert "Understood" in result.response_text or "How" in result.response_text


@pytest.mark.asyncio
async def test_cancellation_confirmation_fast_path_unchanged_by_rag_draft_suppression():
    """Phase 5B-2: Pending cancellation confirmation fast path remains completely intact."""
    mock_tp = MockToolProvider(should_succeed=True)
    mock_llm = MockLLMProvider()
    engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-rag-empty-draft-5", "+919876543210", domain=DomainType.INFRA_BUILDER)

    # Set up pending appointment cancellation state
    state.set_pending_action("cancel_calendar_event")
    state.update_slot("cancellation_confirmation_pending_id", 42)
    state.update_slot("cancellation_confirmation_pending_details", "tomorrow at 10:00 AM")
    state.update_slot("returning_caller_verified_sensitive", True)

    result = await engine.process_user_turn(state, "yes, cancel it please")

    # Confirmation fast path executes tool without invoking decision LLM
    assert len(mock_tp.executed_tools) == 1
    assert mock_tp.executed_tools[0].tool_name == "cancel_calendar_event"
    assert mock_tp.executed_tools[0].arguments["appointment_id"] == 42
    assert result.tool_result is not None
    assert result.tool_result.success is True
    assert "cancelled" in result.response_text.lower()
    assert len(mock_llm.invocations) == 0, "Confirmation fast path must bypass decision LLM entirely"
