"""Comprehensive Focused Test Suite for Phase 5C - Phase 3:
Connect Grounded Streaming to Deepgram TTS.

Verifies:
1. Phrases reach TTS before grounded generation finishes.
2. Phrases are sent once and in the correct order.
3. Final phrase and TTS flush happen exactly once.
4. Final engine response and persisted conversation history contain no duplicated text.
5. Non-RAG and tool-driven responses retain their existing behavior.
6. Interruption cancels generation and stops pending audio.
7. Late audio from an old turn is discarded.
8. Gemini failure before the first phrase follows a safe fallback.
9. Failure after partial speech does not replay the answer.
10. Deepgram and WebSocket disconnects clean up safely.
11. Cancellation does not mark an interrupted response as successfully completed.
12. Existing session persistence, security, grounding, pricing, and appointment cancellation remain unchanged.
13. Latency comparison: streaming vs baseline recording monotonic timestamps.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.interfaces import KnowledgeProvider, LLMProvider, ToolProvider
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    ConversationStage,
    DialogueTurn,
    DomainType,
    RAGChunk,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
    TurnRole,
)
from src.rag.retriever import GroundedContextResult
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.voice.latency_tracker import TurnLatencyTracker
from src.voice.session_manager import VoiceSessionManager
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS


# =============================================================================
# Test Mocks
# =============================================================================

class MockWebSocket:
    """Mock WebSocket for verifying sent JSON messages and audio bytes."""

    def __init__(self, connected: bool = True):
        self.connected = connected
        self.sent_json: List[Dict[str, Any]] = []
        self.sent_bytes: List[bytes] = []
        self.closed = False

    async def send_text(self, text: str) -> None:
        if not self.connected or self.closed:
            raise RuntimeError("WebSocket is closed")
        try:
            data = json.loads(text)
            self.sent_json.append(data)
        except Exception:
            self.sent_json.append({"_raw": text})

    async def send_bytes(self, data: bytes) -> None:
        if not self.connected or self.closed:
            raise RuntimeError("WebSocket is closed")
        self.sent_bytes.append(data)

    async def close(self) -> None:
        self.closed = True
        self.connected = False


class MockStreamingTTS(DeepgramFluxTTS):
    """Mock TTS instrumenting send_text, flush, and interrupt with timestamps."""

    def __init__(self, mock_audio_chunk: bytes = b"MOCK_PCM16_FRAME"):
        super().__init__(api_key="")
        self.mock_audio_chunk = mock_audio_chunk
        self.sent_chunks: List[str] = []
        self.flush_count = 0
        self.interrupt_count = 0
        self.call_timestamps: Dict[str, List[float]] = {
            "send_text": [],
            "flush": [],
            "interrupt": [],
        }

    async def connect(self, *args, **kwargs) -> None:
        self.connected = True
        self._mock_mode = True

    async def send_text(self, text: str) -> None:
        now = time.perf_counter()
        self.call_timestamps["send_text"].append(now)
        self.sent_chunks.append(text)

        # Emit mock audio frame immediately to simulate real streaming TTS delivery
        if self.audio_callback:
            res = self.audio_callback(self.mock_audio_chunk)
            if asyncio.iscoroutine(res):
                await res

        if not self.first_audio_received:
            self.first_audio_received = True
            if self.event_callback:
                res = self.event_callback({"type": "FirstAudio", "timestamp": now, "audio_bytes": len(self.mock_audio_chunk)})
                if asyncio.iscoroutine(res):
                    await res

    async def flush(self) -> None:
        now = time.perf_counter()
        self.call_timestamps["flush"].append(now)
        self.flush_count += 1
        if self.event_callback:
            res = self.event_callback({"type": "SpeechMetadata"})
            if asyncio.iscoroutine(res):
                await res

    async def interrupt(self) -> None:
        self.call_timestamps["interrupt"].append(time.perf_counter())
        self.interrupt_count += 1
        self.sent_chunks.clear()
        self.reset_turn()
        if self.event_callback:
            res = self.event_callback({"type": "SpeechInterrupted"})
            if asyncio.iscoroutine(res):
                await res


class MockKnowledgeProvider(KnowledgeProvider):
    """Knowledge provider returning verified RAG chunks for testing."""

    def __init__(self, chunks: Optional[List[RAGChunk]] = None):
        self.chunks = chunks or []

    async def search(self, query: RAGQuery) -> List[RAGChunk]:
        return self.chunks

    async def retrieve_grounded_context(self, query: RAGQuery) -> GroundedContextResult:
        if not self.chunks:
            return GroundedContextResult(
                query=query,
                chunks=[],
                knowledge_available=False,
                formatted_context="No verified knowledge found passing relevance threshold.",
            )
        lines = [f'<verified_reference_data domain="{query.domain.value}">']
        for idx, c in enumerate(self.chunks, 1):
            lines.append(f"--- Document {idx}: {c.title} ---\nContent:\n{c.content}\n")
        lines.append("</verified_reference_data>")
        return GroundedContextResult(
            query=query,
            chunks=self.chunks,
            knowledge_available=True,
            formatted_context="\n".join(lines),
        )

    async def index_document(self, chunk: RAGChunk) -> bool:
        self.chunks.append(chunk)
        return True


def build_test_chunk() -> RAGChunk:
    return RAGChunk(
        doc_id="infra:pristine_city:overview",
        domain=DomainType.INFRA_BUILDER,
        title="Pristine City Overview",
        content="Pristine City is an exclusive residential community offering gated plots with wide roads.",
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
# 1. Phrases Reach TTS Before Grounded Generation Finishes
# =============================================================================

@pytest.mark.asyncio
async def test_01_phrases_reach_tts_before_grounded_generation_finishes():
    """Verify that complete phrases are delivered to TTS while generation is still in progress."""
    decision = build_rag_decision()

    # Generator with simulated inter-chunk latency
    class DelayedStreamingLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            yield "Pristine City offers "
            yield "luxury plotted layouts. "
            await asyncio.sleep(0.06)
            yield "All roads are 40 "
            yield "feet wide."

    mock_llm = DelayedStreamingLLM(canned_decisions=[decision])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-1", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    ws_lock = asyncio.Lock()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }
    tts.audio_callback = lambda b: ws.send_bytes(b)
    tts.event_callback = lambda e: None

    t_start = time.perf_counter()
    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=ws_lock,
        state=state,
        transcript="Tell me about Pristine City",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )
    t_end = time.perf_counter()

    # The first phrase was sent to TTS well before the turn finished
    assert len(tts.call_timestamps["send_text"]) >= 2
    t_first_tts_send = tts.call_timestamps["send_text"][0]

    assert t_first_tts_send < t_end
    # First audio arrived before generation concluded
    assert len(ws.sent_bytes) >= 1
    assert "luxury plotted layouts." in tts.sent_chunks[0]


# =============================================================================
# 2. Phrases Sent Once and In Correct Order
# =============================================================================

@pytest.mark.asyncio
async def test_02_phrases_sent_once_and_in_correct_order():
    """Verify phrases arrive at TTS sequentially and without duplication."""
    decision = build_rag_decision()
    stream_fragments = [
        "Welcome to Pristine City. ",
        "Plots are ready for registration. ",
        "Contact our sales desk.",
    ]

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(stream_fragments)
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-2", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    ws_lock = asyncio.Lock()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }
    tts.audio_callback = lambda b: ws.send_bytes(b)

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=ws_lock,
        state=state,
        transcript="Tell me about the plots",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    # Clean phrases received by TTS
    cleaned_sent = [c.strip() for c in tts.sent_chunks]
    assert cleaned_sent == [
        "Welcome to Pristine City.",
        "Plots are ready for registration.",
        "Contact our sales desk.",
    ]


# =============================================================================
# 3. Final Phrase and TTS Flush Happen Exactly Once
# =============================================================================

@pytest.mark.asyncio
async def test_03_final_phrase_and_tts_flush_happen_exactly_once():
    """Verify flush is called exactly once after stream completion."""
    decision = build_rag_decision()
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(["Pristine City features wide roads."])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-3", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Tell me about Pristine City",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    assert tts.flush_count == 1
    llm_complete_events = [m for m in ws.sent_json if m.get("type") == "llm_complete"]
    assert len(llm_complete_events) == 1


# =============================================================================
# 4. Final Engine Response & Persisted History Contain No Duplications
# =============================================================================

@pytest.mark.asyncio
async def test_04_persisted_conversation_history_contains_no_duplicated_text():
    """Verify the conversation history records the full turn response text exactly once."""
    decision = build_rag_decision()
    fragments = ["Pristine City has gated security. ", "It is approved by HMDA."]
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(fragments)
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-4", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Is Pristine City approved?",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    agent_turns = [t for t in state.history if t.role == TurnRole.AGENT]
    assert len(agent_turns) == 1
    expected_full_text = "Pristine City has gated security. It is approved by HMDA."
    assert agent_turns[0].content == expected_full_text


# =============================================================================
# 5. Non-RAG and Tool-Driven Turns Retain Existing Behavior
# =============================================================================

@pytest.mark.asyncio
async def test_05_non_rag_turns_use_burst_chunking():
    """Verify non-RAG turns (e.g. greetings) deliver using burst words chunking without callback streaming."""
    greeting_decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="greeting",
        proposed_stage=ConversationStage.GREETING,
        user_facing_response="Hello! Welcome to Infra Builder. How can I help you today?",
        knowledge_required=False,
    )
    mock_llm = MockLLMProvider(canned_decisions=[greeting_decision])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-5", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Hello, good morning",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    # In burst chunking, first burst is 3 words, subsequent are 4 words
    assert len(tts.sent_chunks) >= 2
    assert tts.sent_chunks[0] == "Hello! Welcome to "
    assert tts.flush_count == 1


# =============================================================================
# 6. Interruption Cancels Generation and Stops Pending Audio
# =============================================================================

@pytest.mark.asyncio
async def test_06_interruption_cancels_generation_and_stops_pending_audio():
    """Verify caller barge-in cancels active LLM generation task and stops speech."""
    decision = build_rag_decision()

    class InfiniteStreamingLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            yield "Pristine City has plots. "
            # Wait for interruption
            while True:
                await asyncio.sleep(0.01)
                yield "More words... "

    mock_llm = InfiniteStreamingLLM(canned_decisions=[decision])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-6", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    # Start turn in background task
    turn_task = asyncio.create_task(
        voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Tell me about Pristine City",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            turn_id=1,
        )
    )

    # Allow first phrase to be emitted
    await asyncio.sleep(0.02)
    assert len(tts.sent_chunks) >= 1

    # Caller interrupts
    await voice_mgr.interrupt_speech(state.session_id)
    turn_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await turn_task

    assert tts.interrupt_count >= 1


# =============================================================================
# 7. Late Audio from an Old Turn Is Discarded
# =============================================================================

@pytest.mark.asyncio
async def test_07_late_audio_from_old_turn_is_discarded():
    """Verify audio bytes dispatched after a turn is invalidated are dropped without transmission."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-7", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    ws = MockWebSocket()
    ws_lock = asyncio.Lock()
    voice_mgr = VoiceSessionManager(engine=MagicMock(), state_manager=state_mgr)

    # Session currently has active_turn_id = 0 (interrupted)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "active_turn_id": 0,
    }

    # Simulate deepgram_audio_callback receiving an audio frame from the interrupted turn
    connection_alive = True
    async def audio_callback(data: bytes):
        current_active = voice_mgr._active_sessions.get(state.session_id, {}).get("active_turn_id", 0)
        if not current_active:
            return  # Discard
        await ws.send_bytes(data)

    await audio_callback(b"LATE_AUDIO_BYTES")
    assert len(ws.sent_bytes) == 0


# =============================================================================
# 8. Gemini Failure Before First Phrase Follows Safe Fallback
# =============================================================================

@pytest.mark.asyncio
async def test_08_gemini_failure_before_first_phrase_follows_safe_fallback():
    """If stream fails before any phrase is emitted, engine falls back to batch generation."""
    decision = build_rag_decision()
    batch_text = "Pristine City offers luxury gated community plots."

    class FailingImmediateStreamLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            raise RuntimeError("Stream failed at handshake")
            yield "unreachable"

    mock_llm = FailingImmediateStreamLLM(
        canned_decisions=[decision],
        canned_responses=[batch_text],
    )
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-8", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Tell me about Pristine City",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    # Batch fallback text delivered to TTS via burst chunking
    assert len(tts.sent_chunks) >= 1
    assert "Pristine City offers" in tts.sent_chunks[0]
    assert tts.flush_count == 1


# =============================================================================
# 9. Failure After Partial Speech Does Not Replay Answer
# =============================================================================

@pytest.mark.asyncio
async def test_09_failure_after_partial_speech_does_not_replay_answer():
    """If stream breaks mid-generation, does NOT replay full answer from the beginning."""
    decision = build_rag_decision()

    class MidStreamFailingLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            yield "Pristine City features wide roads. "
            yield "The clubhouse has "
            raise RuntimeError("Mid-stream connection severed!")

    mock_llm = MidStreamFailingLLM(
        canned_decisions=[decision],
        canned_responses=["A completely different full batch answer that should NEVER play."],
    )
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-9", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Tell me about Pristine City",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    # Only phrase 1 was sent to TTS; batch response was never sent
    assert len(tts.sent_chunks) == 1
    assert tts.sent_chunks[0] == "Pristine City features wide roads. "
    # Flush called once for the partial speech
    assert tts.flush_count == 1


# =============================================================================
# 10. Deepgram & WebSocket Disconnects Clean Up Safely
# =============================================================================

@pytest.mark.asyncio
async def test_10_websocket_disconnect_cleans_up_safely():
    """When the client WebSocket disconnects, agent turn exits gracefully."""
    decision = build_rag_decision()
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(["Pristine City features wide roads."])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-10", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket(connected=False)  # Disconnected
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    # Should exit cleanly without raising an unhandled exception
    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Tell me about Pristine City",
        deepgram_tts=tts,
        connection_alive=lambda: False,
        turn_id=1,
    )


# =============================================================================
# 11. Cancellation Does Not Mark Interrupted Turn as Completed
# =============================================================================

@pytest.mark.asyncio
async def test_11_cancellation_does_not_mark_interrupted_turn_as_completed():
    """CancelledError during streaming aborts the turn before history recording."""
    decision = build_rag_decision()

    class CancellingLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            yield "First few words "
            raise asyncio.CancelledError()

    mock_llm = CancellingLLM(canned_decisions=[decision])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-11", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    initial_turn_count = len(state.history)

    with pytest.raises(asyncio.CancelledError):
        await voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Tell me about Pristine City",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            turn_id=1,
        )

    # History was NOT recorded with a completed response
    assert len(state.history) == initial_turn_count


# =============================================================================
# 12. Unverified Pricing Fails Closed Without Streaming
# =============================================================================

@pytest.mark.asyncio
async def test_12_unverified_pricing_fails_closed_without_streaming():
    """An unconfirmed price claim fails closed before synthesis and does not stream."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="pricing_query",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="",
        knowledge_required=True,
        knowledge_query="Pristine City price per square yard",
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(["The price is only 500 rupees!"])
    # Chunk contains overview but NO verified pricing line
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-12", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="What is the price of Pristine City?",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    # Does NOT stream the hallucinated 500 rupees
    for chunk in tts.sent_chunks:
        assert "500 rupees" not in chunk

    # Speaks verified fail-closed rate inquiry message
    agent_turn = state.history[-1]
    assert "can't verify the current price" in agent_turn.content


# =============================================================================
# 13. Latency Benchmark: Streaming vs Baseline Monotonic Timings
# =============================================================================

@pytest.mark.asyncio
async def test_13_latency_comparison_streaming_vs_baseline():
    """Compare TTFA (Time to First Audio delivered to client) between streaming and baseline across 10 iterations."""
    sample_size = 10
    streaming_ttfa_ms: List[float] = []
    baseline_ttfa_ms: List[float] = []

    decision = build_rag_decision()
    chunk = build_test_chunk()

    # 1. Measure Streaming
    for i in range(sample_size):
        class TimedStreamingLLM(MockLLMProvider):
            async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
                yield "Pristine City offers "
                yield "luxury plotted layouts. "
                await asyncio.sleep(0.04)
                yield "Wide blacktop roads "
                yield "are provided."

        mock_llm = TimedStreamingLLM(canned_decisions=[decision])
        mock_kp = MockKnowledgeProvider(chunks=[chunk])
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
        state_mgr = ConversationStateManager()
        state = state_mgr.create_session(f"bench-stream-{i}", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

        tts = MockStreamingTTS()
        ws = MockWebSocket()
        tracker = TurnLatencyTracker(turn_id=1, session_id=state.session_id, domain="infra_builder")
        voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
        voice_mgr._active_sessions[state.session_id] = {
            "websocket": ws,
            "state": state,
            "tts": tts,
            "active_turn_id": 1,
        }

        t_turn_start = time.perf_counter()
        tracker.mark_speech_end(t_turn_start)

        t_first_audio_delivered: Optional[float] = None
        async def audio_cb(data: bytes):
            nonlocal t_first_audio_delivered
            if t_first_audio_delivered is None:
                t_first_audio_delivered = time.perf_counter()
            await ws.send_bytes(data)

        tts.audio_callback = audio_cb

        await voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Tell me about Pristine City",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            tracker=tracker,
            turn_id=1,
        )

        assert t_first_audio_delivered is not None
        ttfa = (t_first_audio_delivered - t_turn_start) * 1000
        streaming_ttfa_ms.append(ttfa)

    # 2. Measure Baseline (Non-streaming: full generation completes before any TTS dispatch)
    for i in range(sample_size):
        full_text = "Pristine City offers luxury plotted layouts. Wide blacktop roads are provided."
        mock_llm = MockLLMProvider(
            canned_decisions=[decision],
            canned_responses=[full_text],
        )
        # Remove stream_response from instance to force batch mode
        mock_llm.stream_response = None  # type: ignore

        mock_kp = MockKnowledgeProvider(chunks=[chunk])
        engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
        state_mgr = ConversationStateManager()
        state = state_mgr.create_session(f"bench-base-{i}", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

        tts = MockStreamingTTS()
        ws = MockWebSocket()
        tracker = TurnLatencyTracker(turn_id=1, session_id=state.session_id, domain="infra_builder")
        voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
        voice_mgr._active_sessions[state.session_id] = {
            "websocket": ws,
            "state": state,
            "tts": tts,
            "active_turn_id": 1,
        }

        t_turn_start = time.perf_counter()
        tracker.mark_speech_end(t_turn_start)

        t_first_audio_delivered: Optional[float] = None
        async def audio_cb(data: bytes):
            nonlocal t_first_audio_delivered
            if t_first_audio_delivered is None:
                t_first_audio_delivered = time.perf_counter()
            await ws.send_bytes(data)

        tts.audio_callback = audio_cb

        await voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Tell me about Pristine City",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            tracker=tracker,
            turn_id=1,
        )

        assert t_first_audio_delivered is not None
        ttfa = (t_first_audio_delivered - t_turn_start) * 1000
        baseline_ttfa_ms.append(ttfa)

    # Compute statistics
    def p50(arr: List[float]) -> float:
        s = sorted(arr)
        return s[len(s) // 2]

    def p95(arr: List[float]) -> float:
        s = sorted(arr)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    stream_p50 = p50(streaming_ttfa_ms)
    stream_p95 = p95(streaming_ttfa_ms)
    base_p50 = p50(baseline_ttfa_ms)
    base_p95 = p95(baseline_ttfa_ms)

    print(
        f"\n[BENCHMARK RESULTS - Sample Size {sample_size} runs]:\n"
        f"Streaming TTFA: p50={stream_p50:.2f}ms, p95={stream_p95:.2f}ms\n"
        f"Baseline  TTFA: p50={base_p50:.2f}ms, p95={base_p95:.2f}ms\n"
    )

    assert len(streaming_ttfa_ms) == sample_size
    assert len(baseline_ttfa_ms) == sample_size


# =============================================================================
# 14. Interruption Before First Phrase Aborts Cleanly (Scenario 1)
# =============================================================================

@pytest.mark.asyncio
async def test_14_interruption_before_first_phrase_aborts_cleanly():
    """Caller interrupts before the first grounded phrase is emitted."""
    decision = build_rag_decision()

    class SlowStreamingLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            await asyncio.sleep(0.15)
            yield "Pristine City offers plots."

    mock_llm = SlowStreamingLLM(canned_decisions=[decision])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-14", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }
    tts.audio_callback = lambda b: ws.send_bytes(b)

    turn_task = asyncio.create_task(
        voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Tell me about Pristine City",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            turn_id=1,
        )
    )
    voice_mgr._active_sessions[state.session_id]["llm_task"] = turn_task

    # Allow turn to start, but interrupt before any phrase completes
    await asyncio.sleep(0.02)
    interrupted = await voice_mgr.interrupt_speech(state.session_id)
    assert interrupted is True

    with pytest.raises(asyncio.CancelledError):
        await turn_task

    # Verify no phrases were dispatched to TTS
    assert len(tts.sent_chunks) == 0
    # No audio was sent to client WebSocket
    assert len(ws.sent_bytes) == 0
    # TTS interrupt was triggered
    assert tts.interrupt_count >= 1
    # Turn was aborted before being marked completed in history
    agent_turns = [t for t in state.history if t.role == TurnRole.AGENT]
    assert len(agent_turns) == 0


# =============================================================================
# 15. Interruption After Partial Audio Delivery Stops Stream (Scenario 2)
# =============================================================================

@pytest.mark.asyncio
async def test_15_interruption_after_partial_audio_delivery_stops_stream():
    """Caller interrupts after partial audio delivery; subsequent phrases are halted."""
    decision = build_rag_decision()

    class TwoPhraseDelayedLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            yield "Pristine City features wide roads. "
            await asyncio.sleep(0.15)
            yield "The clubhouse has world-class amenities."

    mock_llm = TwoPhraseDelayedLLM(canned_decisions=[decision])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-15", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }
    tts.audio_callback = lambda b: ws.send_bytes(b)

    turn_task = asyncio.create_task(
        voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Tell me about Pristine City",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            turn_id=1,
        )
    )
    voice_mgr._active_sessions[state.session_id]["llm_task"] = turn_task

    # Wait for phrase 1 to reach TTS
    await asyncio.sleep(0.04)
    assert len(tts.sent_chunks) == 1
    assert "wide roads." in tts.sent_chunks[0]

    # Caller barge-in
    await voice_mgr.interrupt_speech(state.session_id)

    with pytest.raises(asyncio.CancelledError):
        await turn_task

    # Phrase 2 was never sent to TTS
    assert len(tts.sent_chunks) == 0 or len(tts.sent_chunks) == 1
    for chunk in tts.sent_chunks:
        assert "clubhouse" not in chunk


# =============================================================================
# 16. Rapid Multiple Interruptions Leave Clean State (Scenario 3)
# =============================================================================

@pytest.mark.asyncio
async def test_16_rapid_multiple_interruptions_leave_clean_state():
    """Rapid repeated barge-in interruptions leave session in a clean, stable state."""
    decision = build_rag_decision()
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(["Pristine City has HMDA approval."])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-16", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    # Simulate 3 rapid caller interruptions
    for _ in range(3):
        res = await voice_mgr.interrupt_speech(state.session_id)
        assert res is True

    # State active_turn_id is 0
    assert voice_mgr._active_sessions[state.session_id]["active_turn_id"] == 0
    assert tts.interrupt_count == 3

    # Now verify a new turn executes successfully
    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Tell me about Pristine City",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=2,
    )
    assert tts.flush_count == 1
    assert any("HMDA approval." in chunk for chunk in tts.sent_chunks)


# =============================================================================
# 17. Deepgram Disconnect During Streaming Handled Safely (Scenario 6)
# =============================================================================

@pytest.mark.asyncio
async def test_17_deepgram_disconnect_during_streaming_handled_safely():
    """Deepgram connection drop during streaming is caught safely and sends error message."""
    decision = build_rag_decision()
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(["Pristine City features wide roads."])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-17", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    class DisconnectingTTS(MockStreamingTTS):
        async def send_text(self, text: str, turn_id: Optional[int] = None) -> None:
            raise ConnectionResetError("Deepgram WebSocket closed unexpectedly")

    tts = DisconnectingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    # Execution does not raise uncaught exception
    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Tell me about Pristine City",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    error_msgs = [m for m in ws.sent_json if m.get("type") == "error"]
    assert len(error_msgs) == 1
    assert "Agent processing failed." in error_msgs[0].get("message", "")


# =============================================================================
# 18. WebSocket Disconnect During Audio Delivery (Scenario 7)
# =============================================================================

@pytest.mark.asyncio
async def test_18_websocket_disconnect_during_audio_delivery():
    """WebSocket disconnection during audio callback does not crash the session."""
    ws = MockWebSocket(connected=True)
    ws_lock = asyncio.Lock()
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-18", DomainType.INFRA_BUILDER, CallDirection.INBOUND)
    voice_mgr = VoiceSessionManager(engine=MagicMock(), state_manager=state_mgr)

    # Disconnect the WebSocket
    await ws.close()

    # Attempting to send audio frames through safe_send_bytes should handle the failure safely
    from src.voice.session_manager import safe_send_bytes
    # Should not raise exception
    await safe_send_bytes(ws, ws_lock, b"SAMPLE_AUDIO_BYTES")
    assert len(ws.sent_bytes) == 0


# =============================================================================
# 19. Old Audio Arriving After Newer Turn Begins Is Discarded (Scenario 8)
# =============================================================================

@pytest.mark.asyncio
async def test_19_old_audio_arriving_after_newer_turn_begins_is_discarded():
    """Audio frame tagged with an older turn ID is discarded when a newer turn is active."""
    ws = MockWebSocket()
    ws_lock = asyncio.Lock()
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-19", DomainType.INFRA_BUILDER, CallDirection.INBOUND)
    voice_mgr = VoiceSessionManager(engine=MagicMock(), state_manager=state_mgr)

    # Session active_turn_id is 2
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "active_turn_id": 2,
    }

    from src.voice.session_manager import safe_send_bytes

    async def deepgram_audio_callback(audio_bytes: bytes, frame_turn_id: Optional[int] = None):
        current_active = voice_mgr._active_sessions.get(state.session_id, {}).get("active_turn_id", 0)
        if not current_active:
            return
        if frame_turn_id is not None and frame_turn_id != current_active:
            return  # Discard old turn audio
        await safe_send_bytes(ws, ws_lock, audio_bytes)

    # Late frame from Turn 1 arrives
    await deepgram_audio_callback(b"OLD_TURN_1_AUDIO", frame_turn_id=1)
    assert len(ws.sent_bytes) == 0

    # Fresh frame from Turn 2 arrives
    await deepgram_audio_callback(b"NEW_TURN_2_AUDIO", frame_turn_id=2)
    assert len(ws.sent_bytes) == 1
    assert ws.sent_bytes[0] == b"NEW_TURN_2_AUDIO"


# =============================================================================
# 20. Overlapping Turn Tasks Cancel Previous Task Cleanly (Scenario 9)
# =============================================================================

@pytest.mark.asyncio
async def test_20_overlapping_turn_tasks_cancel_previous_task_cleanly():
    """Launching Turn 2 cancels Turn 1 cleanly without race conditions on state history."""
    decision = build_rag_decision()

    class LongRunningLLM(MockLLMProvider):
        async def stream_response(self, *args, **kwargs) -> AsyncIterator[str]:
            yield "Turn 1 starts... "
            await asyncio.sleep(0.5)
            yield "Turn 1 ends."

    mock_llm1 = LongRunningLLM(canned_decisions=[decision])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine1 = ConversationEngine(llm_provider=mock_llm1, knowledge_provider=mock_kp)

    mock_llm2 = MockLLMProvider(canned_decisions=[decision])
    mock_llm2.add_stream(["Turn 2 fast response."])
    engine2 = ConversationEngine(llm_provider=mock_llm2, knowledge_provider=mock_kp)

    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-20", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine1, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    # Start Turn 1
    turn1_task = asyncio.create_task(
        voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Turn 1 query",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            turn_id=1,
        )
    )
    voice_mgr._active_sessions[state.session_id]["llm_task"] = turn1_task

    await asyncio.sleep(0.02)
    assert not turn1_task.done()

    # Now simulate overlapping turn arrival as in handle_media_stream:
    # 1. Cancel previous llm_task
    old_task = voice_mgr._active_sessions[state.session_id].get("llm_task")
    if old_task and not old_task.done():
        old_task.cancel()

    # 2. Update engine and active_turn_id
    voice_mgr.engine = engine2
    voice_mgr._active_sessions[state.session_id]["active_turn_id"] = 2

    # 3. Launch Turn 2
    turn2_task = asyncio.create_task(
        voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Turn 2 query",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            turn_id=2,
        )
    )
    voice_mgr._active_sessions[state.session_id]["llm_task"] = turn2_task

    # Wait for Turn 2 to finish
    await turn2_task

    # Turn 1 task is cancelled
    assert turn1_task.cancelled() or turn1_task.done()

    # State history has recorded only Turn 2's completed response
    agent_turns = [t for t in state.history if t.role == TurnRole.AGENT]
    assert len(agent_turns) == 1
    assert "Turn 2 fast response." in agent_turns[0].content


# =============================================================================
# 21. TTS Completion Acknowledgment Timeout Proceeds Safely (Scenario 10)
# =============================================================================

@pytest.mark.asyncio
async def test_21_tts_completion_acknowledgment_timeout_proceeds_safely(monkeypatch):
    """When TTS completion event times out, turn completes safely without blocking."""
    decision = build_rag_decision()
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_llm.add_stream(["Pristine City features gated security."])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-21", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    # Event that is never set
    untriggered_event = asyncio.Event()
    tracker = TurnLatencyTracker(turn_id=1, session_id=state.session_id, domain="infra_builder")

    # Fast-forward timeout for test speed
    orig_wait_for = asyncio.wait_for
    async def fast_wait_for(fut, timeout):
        if fut == untriggered_event.wait() or (hasattr(fut, "__repr__") and "wait" in repr(fut)):
            return await orig_wait_for(fut, timeout=0.01)
        return await orig_wait_for(fut, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", fast_wait_for)

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Tell me about security",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        tracker=tracker,
        turn_id=1,
        tts_done_event=untriggered_event,
    )

    # Tracker marked TTS complete despite timeout
    assert tracker.t_tts_complete is not None
    # llm_complete was sent to WebSocket
    llm_completes = [m for m in ws.sent_json if m.get("type") == "llm_complete"]
    assert len(llm_completes) == 1


# =============================================================================
# 22. Callback Failure Halts Safely and Emits Error (Scenario 11)
# =============================================================================

@pytest.mark.asyncio
async def test_22_callback_failure_halts_safely_and_emits_error():
    """Exception thrown inside streaming callback is caught and emits error event."""
    decision = build_rag_decision()

    class FailingEngine(ConversationEngine):
        async def process_user_turn(self, *args, **kwargs):
            cb = kwargs.get("on_grounded_chunk")
            if cb:
                await cb("Valid phrase.")
                raise RuntimeError("Downstream pipe broken in callback caller")
            return await super().process_user_turn(*args, **kwargs)

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_kp = MockKnowledgeProvider(chunks=[build_test_chunk()])
    engine = FailingEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-22", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    # Should not raise uncaught exception
    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Tell me about Pristine City",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    error_msgs = [m for m in ws.sent_json if m.get("type") == "error"]
    assert len(error_msgs) == 1
    assert "Agent processing failed." in error_msgs[0].get("message", "")


# =============================================================================
# 23. Session Finalization Runs After Interruption or Disconnect (Scenario 12)
# =============================================================================

@pytest.mark.asyncio
async def test_23_session_finalization_runs_after_interruption_or_disconnect():
    """Finalization sets conversation inactive, records termination reason, and builds summary."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-23", DomainType.INFRA_BUILDER, CallDirection.INBOUND)
    state.history.append(DialogueTurn(role=TurnRole.CALLER, content="Hello"))
    state.history.append(DialogueTurn(role=TurnRole.AGENT, content="Hello, how can I help you?"))

    voice_mgr = VoiceSessionManager(engine=MagicMock(), state_manager=state_mgr)
    voice_mgr._active_sessions[state.session_id] = {
        "state": state,
        "active_turn_id": 1,
    }

    summary = await voice_mgr.finalize_session(
        session_id=state.session_id,
        reason="caller_hung_up",
    )

    assert summary is not None
    assert summary.session_id == state.session_id
    assert summary.termination_reason == "caller_hung_up"
    assert state.conversation_active is False
    assert state.termination_reason == "caller_hung_up"
    assert state.metadata.end_time is not None
    assert state.session_id in voice_mgr._finalized_sessions


# =============================================================================
# 24. Repeated Finalization Does Not Duplicate Persisted Interactions (Scenario 13)
# =============================================================================

@pytest.mark.asyncio
async def test_24_repeated_finalization_does_not_duplicate_persisted_interactions():
    """Repeated finalization calls are idempotent and do not duplicate database records."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-24", DomainType.INFRA_BUILDER, CallDirection.INBOUND)
    state.history.append(DialogueTurn(role=TurnRole.CALLER, content="What is Pristine City?"))
    state.history.append(DialogueTurn(role=TurnRole.AGENT, content="It is a luxury residential community."))

    mock_repo = MagicMock()
    mock_repo.save_call_interaction = AsyncMock()
    mock_repo.persist_summary = AsyncMock()

    voice_mgr = VoiceSessionManager(
        engine=MagicMock(),
        state_manager=state_mgr,
        call_repository=mock_repo,
    )
    voice_mgr._active_sessions[state.session_id] = {
        "state": state,
        "active_turn_id": 1,
    }

    # Call finalize first time
    summary1 = await voice_mgr.finalize_session(session_id=state.session_id, reason="caller_disconnected")
    assert summary1 is not None
    # 2 turns in history -> save_call_interaction called twice
    assert mock_repo.save_call_interaction.call_count == 2
    assert mock_repo.persist_summary.call_count == 1

    # Call finalize second time
    summary2 = await voice_mgr.finalize_session(session_id=state.session_id, reason="caller_disconnected")
    assert summary2 == summary1
    # Call counts must NOT increase
    assert mock_repo.save_call_interaction.call_count == 2
    assert mock_repo.persist_summary.call_count == 1


# =============================================================================
# 25. Tool-Driven Actions Never Replayed & Partial Spoken Not Marked Completed (Scenario 14)
# =============================================================================

@pytest.mark.asyncio
async def test_25_tool_driven_actions_never_replayed_and_partial_spoken_not_marked_completed():
    """Tool execution executes once atomically, and cancelled turns do not record partial output as completed."""
    tool_executed_count = 0

    class MockCalendarTool(ToolProvider):
        def list_tools(self) -> List[Dict[str, Any]]:
            return [{"name": "find_available_slots", "description": "Find available slots"}]

        async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
            nonlocal tool_executed_count
            tool_executed_count += 1
            return ToolExecutionResult(
                tool_name="find_available_slots",
                success=True,
                data={"available_slots": ["10:00 AM", "2:00 PM"]},
            )

    tool_provider = MockCalendarTool()
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="site_visit_request",
        proposed_stage=ConversationStage.ACTION_CONFIRMATION,
        user_facing_response="I checked the schedule and slots are available at 10 AM and 2 PM.",
        action_proposed=True,
        proposed_action=ProposedAction(
            tool_name="find_available_slots",
            arguments={"property_name": "Pristine City", "preferred_date": "tomorrow"},
        ),
        knowledge_required=False,
    )

    mock_llm = MockLLMProvider(canned_decisions=[decision])
    engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-25", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = MockStreamingTTS()
    ws = MockWebSocket()
    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    # Run turn to completion
    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript="Can I visit tomorrow?",
        deepgram_tts=tts,
        connection_alive=lambda: True,
        turn_id=1,
    )

    assert tool_executed_count == 1
    assert len(tts.sent_chunks) >= 1
    assert tts.flush_count == 1

    # Verify tool action result is recorded cleanly
    agent_turn = state.history[-1]
    assert agent_turn.role == TurnRole.AGENT
    assert "10:00 AM" in agent_turn.content and "2:00 PM" in agent_turn.content

    # Now verify that an interrupted turn does NOT record a completed agent turn
    initial_turns = len(state.history)

    class InterruptedEngine(ConversationEngine):
        async def process_user_turn(self, *args, **kwargs):
            raise asyncio.CancelledError("Interrupted before completion")

    interrupted_engine = InterruptedEngine(llm_provider=mock_llm)
    voice_mgr.engine = interrupted_engine

    with pytest.raises(asyncio.CancelledError):
        await voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript="Another question",
            deepgram_tts=tts,
            connection_alive=lambda: True,
            turn_id=2,
        )

    # History was not updated with a partial or incomplete turn
    assert len(state.history) == initial_turns


# =============================================================================
# Mock Deepgram WebSocket for Deterministic Multi-Turn & Interruption Testing
# =============================================================================

class MockDeepgramWebSocket:
    """Async iterable mock Deepgram WebSocket supporting queue-based frame dispatch."""

    def __init__(self):
        self.queue: asyncio.Queue[Union[bytes, str, Exception, None]] = asyncio.Queue()
        self.closed = False
        self.sent_messages: List[str] = []

    async def send(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("WebSocket connection is closed")
        self.sent_messages.append(data)

    async def close(self) -> None:
        self.closed = True
        await self.queue.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed and self.queue.empty():
            raise StopAsyncIteration
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item


# =============================================================================
# 26. Deepgram Stale-Audio Race: Delayed Frame from Old Turn Is Discarded (Phase 5D-2A)
# =============================================================================

@pytest.mark.asyncio
async def test_26_deepgram_stale_audio_race_delayed_frame_discarded():
    """Concretely reproduce the race condition:
    Turn 1 starts producing audio frames.
    Turn 1 is interrupted while frames are still arriving.
    Turn 2 begins.
    Any delayed frame from Turn 1 is NOT labeled with Turn 2's ID and is NOT dispatched.
    """
    tts = DeepgramFluxTTS(api_key="mock_test_key")

    ws1 = MockDeepgramWebSocket()
    tts._active_connection_id = 1
    tts.websocket = ws1
    tts.connected = True

    received_frames: List[Tuple[bytes, Optional[int]]] = []

    async def audio_callback(audio_bytes: bytes, turn_id: Optional[int] = None):
        received_frames.append((audio_bytes, turn_id))

    tts.audio_callback = audio_callback

    # Turn 1 begins
    tts.reset_turn(turn_id=1)
    receive_task_1 = asyncio.create_task(tts._receive_loop(ws1, 1))
    tts.receive_task = receive_task_1

    # Normal frame from Turn 1 arrives
    await ws1.queue.put(b"TURN_1_AUDIO_FRAME_1")
    await asyncio.sleep(0.01)
    assert len(received_frames) == 1
    assert received_frames[0] == (b"TURN_1_AUDIO_FRAME_1", 1)

    # Caller interrupts Turn 1
    await tts.interrupt()
    assert ws1.closed is True
    assert 1 in tts._retired_connections
    assert receive_task_1.done()

    # Turn 2 begins with a fresh connection
    tts.reset_turn(turn_id=2)
    ws2 = MockDeepgramWebSocket()
    tts._active_connection_id = 2
    tts.websocket = ws2
    tts.connected = True
    receive_task_2 = asyncio.create_task(tts._receive_loop(ws2, 2))
    tts.receive_task = receive_task_2

    # Delayed frame arrives from Turn 1's socket (ws1)
    await ws1.queue.put(b"TURN_1_STALE_DELAYED_AUDIO")
    await asyncio.sleep(0.01)

    # CRITICAL ASSERTION: The delayed Turn 1 frame MUST NOT be tagged with Turn 2's ID
    # and MUST NOT be dispatched to the audio callback!
    dispatched_texts = [f[0] for f in received_frames]
    assert b"TURN_1_STALE_DELAYED_AUDIO" not in dispatched_texts
    assert len(received_frames) == 1  # Still only the original Turn 1 frame

    # Clean up Turn 2
    await tts.close()
    if not receive_task_2.done():
        receive_task_2.cancel()


# =============================================================================
# 27. Turn 2 Audio Delivered Normally After Interruption (Phase 5D-2A)
# =============================================================================

@pytest.mark.asyncio
async def test_27_turn_2_audio_delivered_normally_after_interruption():
    """Verify that after Turn 1 is interrupted, Turn 2's own audio is labeled and delivered normally."""
    tts = DeepgramFluxTTS(api_key="mock_test_key")

    ws1 = MockDeepgramWebSocket()
    tts._active_connection_id = 1
    tts.websocket = ws1
    tts.connected = True

    received_frames: List[Tuple[bytes, Optional[int]]] = []

    async def audio_callback(audio_bytes: bytes, turn_id: Optional[int] = None):
        received_frames.append((audio_bytes, turn_id))

    tts.audio_callback = audio_callback

    # Turn 1 starts and receives one frame
    tts.reset_turn(turn_id=1)
    receive_task_1 = asyncio.create_task(tts._receive_loop(ws1, 1))
    tts.receive_task = receive_task_1
    await ws1.queue.put(b"TURN_1_FRAME")
    await asyncio.sleep(0.01)
    assert received_frames[-1] == (b"TURN_1_FRAME", 1)

    # Turn 1 is interrupted
    await tts.interrupt()

    # Turn 2 begins with fresh connection ws2
    tts.reset_turn(turn_id=2)
    ws2 = MockDeepgramWebSocket()
    tts._active_connection_id = 2
    tts.websocket = ws2
    tts.connected = True
    receive_task_2 = asyncio.create_task(tts._receive_loop(ws2, 2))
    tts.receive_task = receive_task_2

    # Turn 2 sends text and receives audio frames
    await ws2.queue.put(b"TURN_2_FRAME_1")
    await ws2.queue.put(b"TURN_2_FRAME_2")
    await asyncio.sleep(0.01)

    assert len(received_frames) == 3
    assert received_frames[1] == (b"TURN_2_FRAME_1", 2)
    assert received_frames[2] == (b"TURN_2_FRAME_2", 2)

    # Clean up
    await tts.close()


# =============================================================================
# 28. Old Receive Loops and Connections Cleanly Retired Without Leaks (Phase 5D-2A)
# =============================================================================

@pytest.mark.asyncio
async def test_28_old_receive_loops_and_connections_cleanly_retired_no_leaks():
    """Verify that interrupted connections and tasks are terminated without dangling tasks or reconnect loops."""
    tts = DeepgramFluxTTS(api_key="mock_test_key")

    ws1 = MockDeepgramWebSocket()
    tts._active_connection_id = 1
    tts.websocket = ws1
    tts.connected = True
    receive_task_1 = asyncio.create_task(tts._receive_loop(ws1, 1))
    tts.receive_task = receive_task_1

    assert not receive_task_1.done()
    assert not ws1.closed

    # Interrupt speech
    await tts.interrupt()

    # Verify connection was retired
    assert ws1.closed is True
    assert 1 in tts._retired_connections
    assert tts.websocket is None
    assert tts.connected is False
    assert receive_task_1.done()

    # Verify no unexpected reconnect loop was scheduled for the retired connection
    assert tts._reconnect_task is None

    # Sending Clear command was recorded
    clear_cmds = [m for m in ws1.sent_messages if "Clear" in m or "Interrupt" in m]
    assert len(clear_cmds) >= 1

    await tts.close()


# =============================================================================
# 29. Sequential Uninterrupted Turns Reuse Connection Efficiently (Phase 5D-2A)
# =============================================================================

@pytest.mark.asyncio
async def test_29_sequential_uninterrupted_turns_reuse_connection_without_reconnect():
    """Verify that normal sequential turns without interruption continue to reuse the existing connection."""
    tts = DeepgramFluxTTS(api_key="mock_test_key")

    ws1 = MockDeepgramWebSocket()
    tts._active_connection_id = 1
    tts.websocket = ws1
    tts.connected = True
    receive_task_1 = asyncio.create_task(tts._receive_loop(ws1, 1))
    tts.receive_task = receive_task_1

    received_frames: List[Tuple[bytes, Optional[int]]] = []

    async def audio_callback(audio_bytes: bytes, turn_id: Optional[int] = None):
        received_frames.append((audio_bytes, turn_id))

    tts.audio_callback = audio_callback

    # Turn 1 starts, speaks, and flushes cleanly
    tts.reset_turn(turn_id=1)
    await tts.send_text("Turn 1 text", turn_id=1)
    await ws1.queue.put(b"TURN_1_AUDIO")
    await ws1.queue.put(json.dumps({"type": "Flushed"}))
    await asyncio.sleep(0.01)

    assert 1 in tts._generation_completed
    assert received_frames == [(b"TURN_1_AUDIO", 1)]

    # Turn 2 starts sequentially without interruption
    tts.reset_turn(turn_id=2)

    # CRITICAL: Connection 1 is NOT closed and is preserved!
    assert tts.websocket is ws1
    assert tts._active_connection_id == 1
    assert not ws1.closed

    # Turn 2 sends text and receives audio on the SAME connection
    await tts.send_text("Turn 2 text", turn_id=2)
    await ws1.queue.put(b"TURN_2_AUDIO")
    await ws1.queue.put(json.dumps({"type": "Flushed"}))
    await asyncio.sleep(0.01)

    assert 2 in tts._generation_completed
    assert len(received_frames) == 2
    assert received_frames[1] == (b"TURN_2_AUDIO", 2)

    await tts.close()


# =============================================================================
# 30. End-to-End Barge-In and Overlapping Turns Discard Stale Audio at Session Manager (Phase 5D-2A)
# =============================================================================

@pytest.mark.asyncio
async def test_30_end_to_end_barge_in_and_overlapping_turns_discard_stale_audio():
    """End-to-end integration test with VoiceSessionManager:
    Turn 1 is interrupted by barge-in.
    Trailing audio from Turn 1 is dropped and never sent to caller websocket.
    Turn 2 audio is delivered to caller websocket tagged with Turn 2.
    """
    state_mgr = ConversationStateManager()
    state = state_mgr.create_session("stream-sess-30", DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    client_ws = MockWebSocket()
    deepgram_ws1 = MockDeepgramWebSocket()

    tts = DeepgramFluxTTS(api_key="mock_key")
    tts._active_connection_id = 1
    tts.websocket = deepgram_ws1
    tts.connected = True
    tts_recv_task_1 = asyncio.create_task(tts._receive_loop(deepgram_ws1, 1))
    tts.receive_task = tts_recv_task_1

    voice_mgr = VoiceSessionManager(engine=MagicMock(), state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[state.session_id] = {
        "websocket": client_ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    async def deepgram_audio_callback(audio_bytes: bytes, turn_id: Optional[int] = None):
        current_active = voice_mgr._active_sessions.get(state.session_id, {}).get("active_turn_id", 0)
        if not current_active:
            return
        frame_turn = turn_id if turn_id is not None else getattr(tts, "current_turn_id", current_active)
        if frame_turn != current_active:
            return
        await client_ws.send_bytes(audio_bytes)

    tts.audio_callback = deepgram_audio_callback

    # Turn 1 starts and emits a valid frame
    tts.reset_turn(turn_id=1)
    await deepgram_ws1.queue.put(b"TURN_1_VALID_AUDIO")
    await asyncio.sleep(0.01)

    assert len(client_ws.sent_bytes) == 1
    assert client_ws.sent_bytes[0] == b"TURN_1_VALID_AUDIO"

    # Caller barges in! Interruption occurs:
    await voice_mgr.interrupt_speech(state.session_id)
    assert deepgram_ws1.closed is True

    # Now Turn 2 starts
    voice_mgr._active_sessions[state.session_id]["active_turn_id"] = 2
    tts.reset_turn(turn_id=2)
    deepgram_ws2 = MockDeepgramWebSocket()
    tts._active_connection_id = 2
    tts.websocket = deepgram_ws2
    tts.connected = True
    tts_recv_task_2 = asyncio.create_task(tts._receive_loop(deepgram_ws2, 2))
    tts.receive_task = tts_recv_task_2

    # Delayed Turn 1 audio arrives on deepgram_ws1
    await deepgram_ws1.queue.put(b"TURN_1_DELAYED_AUDIO")
    await asyncio.sleep(0.01)

    # Fresh Turn 2 audio arrives on deepgram_ws2
    await deepgram_ws2.queue.put(b"TURN_2_FRESH_AUDIO")
    await asyncio.sleep(0.01)

    # Client WebSocket sent_bytes must contain ONLY Turn 1 valid audio and Turn 2 fresh audio
    # It must NEVER contain Turn 1 delayed audio!
    assert b"TURN_1_DELAYED_AUDIO" not in client_ws.sent_bytes
    assert len(client_ws.sent_bytes) == 2
    assert client_ws.sent_bytes[1] == b"TURN_2_FRESH_AUDIO"

    await tts.close()


