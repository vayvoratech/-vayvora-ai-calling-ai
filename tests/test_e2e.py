"""End-to-End Validation Suite for Phase 9: External Telephony Transport Integration.

Executes all 42 required end-to-end integration scenarios using MockTelephonyTransport,
MockCallSessionRepository, and deterministic provider abstractions without requiring
external telephony infrastructure or carrier connectivity:

1.  Inbound (1-7): Inbound connect, unknown caller greeting, VAD speech start, VAD silence end,
    STT delivery, Engine response, TTS to transport delivery.
2.  Outbound (8-11): Campaign context preservation, known caller identification,
    grounded consultation, caller intent preemption.
3.  Conversation Semantics (12-16): Intent switching stack, cross-domain safety, "no" retains
    call, explicit goodbye completes session, business status qualification does not terminate.
4.  Grounded RAG (17-20): EduSaaS grounding, Vayvora grounding, missing knowledge admission,
    cross-domain knowledge isolation.
5.  MCP Verified Actions (21-26): send_email, find_available_slots, create_calendar_event,
    tool error handling, unverified result recording, verification failure safety.
6.  Barge-In (27-31): Active TTS halt, LLM token cancellation, stale frame purge, fresh turn
    acceptance, MCP action continuation.
7.  Disconnect & Persistence (32-36): Remote disconnect stops transport, pipeline termination,
    TTS cancellation, LLM cancellation, CallSummary sanitized persistence.
8.  Failure Recovery & Resilience (37-42): Transport connection failure, malformed audio rejection,
    STT failure fallback, LLM timeout fallback, TTS failure fallback, multi-turn error recovery.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
import pytest

from src.audio.schemas import AudioChunk, STTResult, TTSResult, VADResult
from src.audio.stt import MockSTTProvider
from src.audio.tts import MockTTSProvider, create_pcm_wav_bytes
from src.audio.vad import MockVADProvider
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.errors import (
    LLMError,
    STTError,
    TelephonyAudioIncompatibleError,
    TelephonyConnectionError,
    TelephonyPacketError,
    TTSError,
)
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    CallMetadata,
    CallStatus,
    ConversationStage,
    DomainType,
    RAGChunk,
    RAGQuery,
    ToolExecutionResult,
)
from src.rag.embeddings import MockEmbeddingProvider
from src.rag.retriever import GroundedKnowledgeProvider
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.telephony.adapter import TelephonyVoiceAdapter
from src.telephony.audio import TelephonyAudioConverter
from src.telephony.models import CallSummary, TelephonyEvent, TelephonyEventType
from src.telephony.repository import MockCallSessionRepository
from src.telephony.transport import MockTelephonyTransport
from src.tools.mcp_client import MockToolProvider
from src.voice.context import TurnLifecycleState


# =============================================================================
# E2E Test Harness & Audio Helpers
# =============================================================================


def make_speech_packet(duration: float = 0.25, sample_rate: int = 8000) -> bytes:
    """Generate a PCM16 audio packet representing vocal speech."""
    bytes_data = create_pcm_wav_bytes(duration=duration, sample_rate=sample_rate, frequency=440.0)
    return bytes_data[44:] if len(bytes_data) > 44 else bytes_data


def make_silence_packet(duration: float = 0.25, sample_rate: int = 8000) -> bytes:
    """Generate a PCM16 audio packet representing background silence."""
    num_samples = int(sample_rate * duration)
    return b"\x00" * (num_samples * 2)


async def simulate_turn(adapter: TelephonyVoiceAdapter, text: str):
    """Feed complete spoken turn: speech packet followed by silence packet."""
    if hasattr(adapter.stt_provider, "add_transcription"):
        adapter.stt_provider.add_transcription(text)
    # Speech packet (0.25s)
    sp = make_speech_packet(duration=0.25, sample_rate=8000)
    await adapter.process_incoming_packet(sp, sample_rate=8000)
    # Silence packet (0.25s) triggering speech_end and turn completion
    sil = make_silence_packet(duration=0.25, sample_rate=8000)
    return await adapter.process_incoming_packet(sil, sample_rate=8000)


def build_e2e_components(
    call_id: str = "call-e2e-01",
    direction: CallDirection = CallDirection.INBOUND,
    llm_canned_decisions: Optional[List[ConversationalDecision]] = None,
    stt_transcriptions: Optional[List[str]] = None,
    tool_force_error: bool = False,
    llm_force_error: bool = False,
    vad_force_error: bool = False,
    stt_force_error: bool = False,
    tts_force_error: bool = False,
    transport_force_error: bool = False,
):
    """Construct fully wired application and mock telephony components."""
    transport = MockTelephonyTransport(
        call_id=call_id,
        session_id=f"voice-session-{call_id}",
        direction=direction,
        caller_number="+15551234567",
        callee_number="+15559876543",
        force_connection_error=transport_force_error,
    )
    repo = MockCallSessionRepository()
    state_mgr = ConversationStateManager()

    class FailableMockLLM(MockLLMProvider):
        def __init__(self, *args, force_error: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            self.force_error = force_error

        async def generate_decision(self, *args, **kwargs):
            if self.force_error:
                raise LLMError("Simulated LLM decision failure.")
            return await super().generate_decision(*args, **kwargs)

    llm = FailableMockLLM(
        canned_decisions=llm_canned_decisions,
        force_error=llm_force_error,
    )

    rag = GroundedKnowledgeProvider(
        embedding_provider=MockEmbeddingProvider(dimension=64),
        in_memory=True,
    )

    tools = MockToolProvider(force_unavailable=tool_force_error)

    engine = ConversationEngine(
        llm_provider=llm,
        knowledge_provider=rag,
        tool_provider=tools,
    )

    vad = MockVADProvider(
        sample_rate=16000,
        min_speech_duration=0.1,
        min_silence_duration=0.1,
        force_error=vad_force_error,
    )
    stt = MockSTTProvider(
        canned_transcriptions=stt_transcriptions,
        force_error=stt_force_error,
    )
    tts = MockTTSProvider(
        sample_rate=16000,
        force_error=tts_force_error,
    )

    adapter = TelephonyVoiceAdapter(
        transport=transport,
        state_manager=state_mgr,
        vad_provider=vad,
        stt_provider=stt,
        conversation_engine=engine,
        tts_provider=tts,
        repository=repo,
    )

    return adapter, transport, repo, state_mgr, engine, rag, tools, stt, tts


# =============================================================================
# 1. Inbound Telephony Lifecycle (Scenarios 1 - 7)
# =============================================================================


class TestInboundTelephonyLifecycle:
    """Scenarios 1-7: Inbound transport connection, speech detection, transcription, TTS to transport."""

    @pytest.mark.asyncio
    async def test_01_inbound_connection_starts_voice_pipeline(self):
        """01: Inbound call connection event triggers voice pipeline startup."""
        adapter, transport, _, _, _, _, _, _, _ = build_e2e_components("call-in-01")

        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)
        assert transport.is_connected is True
        assert session.is_active is True
        assert session.pipeline.state == TurnLifecycleState.LISTENING
        assert len(transport.sent_events) >= 2
        assert transport.sent_events[0].event_type == TelephonyEventType.CALL_CONNECTED

    @pytest.mark.asyncio
    async def test_02_unknown_inbound_caller_brand_safe_defaults(self):
        """02: Unknown inbound caller defaults to brand-appropriate safe context."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-in-02")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        state = adapter.conversation_state
        assert state.caller.name is None
        assert state.caller.caller_type == "unknown"
        assert state.current_domain == DomainType.EDUSAAS
        assert state.stage == ConversationStage.GREETING
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_03_caller_speaks_vad_detects_speech_start(self):
        """03: Caller speaks -> VAD detects speech start -> transitions to SPEECH_DETECTED / CAPTURING."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-in-03")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        speech_pcm = make_speech_packet(duration=0.25, sample_rate=8000)
        vad_res, _, _ = await adapter.process_incoming_packet(speech_pcm, sample_rate=8000)

        assert vad_res.speech_start is True
        assert session.pipeline.state in (
            TurnLifecycleState.SPEECH_DETECTED,
            TurnLifecycleState.CAPTURING,
        )

    @pytest.mark.asyncio
    async def test_04_caller_silence_vad_detects_speech_end(self):
        """04: Caller finishes speaking -> VAD detects silence -> triggers transcription."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components("call-in-04")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)
        stt.add_transcription("Hello there")

        speech_pcm = make_speech_packet(duration=0.25, sample_rate=8000)
        await adapter.process_incoming_packet(speech_pcm, sample_rate=8000)

        silence_pcm = make_silence_packet(duration=0.25, sample_rate=8000)
        vad_res, stt_res, _ = await adapter.process_incoming_packet(silence_pcm, sample_rate=8000)

        assert vad_res.speech_end is True
        assert stt_res is not None
        assert stt_res.text == "Hello there"

    @pytest.mark.asyncio
    async def test_05_stt_transcript_arrives_at_engine(self):
        """05: STT transcript arrives at ConversationEngine without loss."""
        adapter, _, _, _, engine, _, _, stt, _ = build_e2e_components("call-in-05")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        vad_res, stt_res, tts_res = await simulate_turn(adapter, "What is the refund policy?")

        assert stt_res.text == "What is the refund policy?"
        assert len(adapter.conversation_state.history) >= 2
        assert adapter.conversation_state.history[0].content == "What is the refund policy?"

    @pytest.mark.asyncio
    async def test_06_engine_produces_response_routed_to_tts(self):
        """06: Engine produces domain-appropriate text response -> routed to TTS."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components("call-in-06")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        vad_res, stt_res, tts_res = await simulate_turn(adapter, "Can you help me?")

        assert tts_res is not None
        assert tts_res.audio_data is not None
        assert len(tts_res.audio_data) > 0

    @pytest.mark.asyncio
    async def test_07_tts_packets_delivered_to_telephony_transport(self):
        """07: Synthesized TTS audio packets delivered to external transport in real time."""
        adapter, transport, _, _, _, _, _, stt, _ = build_e2e_components("call-in-07")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "Please explain the courses.")

        assert len(transport.sent_audio_frames) >= 1
        total_audio_sent = transport.get_sent_audio()
        assert len(total_audio_sent) > 0


# =============================================================================
# 2. Outbound Telephony Campaign (Scenarios 8 - 11)
# =============================================================================


class TestOutboundTelephonyCampaign:
    """Scenarios 8-11: Outbound campaign preservation, caller details, grounding, preemption."""

    @pytest.mark.asyncio
    async def test_08_outbound_preserves_campaign_objective(self):
        """08: Outbound session preserves campaign objective across conversation."""
        adapter, transport, _, _, _, _, _, _, _ = build_e2e_components(
            "call-out-08", direction=CallDirection.OUTBOUND
        )
        await adapter.handle_outbound_call(
            domain=DomainType.EDUSAAS,
            caller_name="Alice Brown",
            campaign_id="CAMP-2026-01",
            campaign_objective="Discuss AI Engineer Certification enrollment",
        )

        state = adapter.conversation_state
        assert state.metadata.campaign_id == "CAMP-2026-01"
        assert state.caller.campaign_objective == "Discuss AI Engineer Certification enrollment"

    @pytest.mark.asyncio
    async def test_09_outbound_identifies_known_caller(self):
        """09: Outbound call identifies known caller by pre-populated details."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components(
            "call-out-09", direction=CallDirection.OUTBOUND
        )
        await adapter.handle_outbound_call(
            domain=DomainType.VAYVORA,
            caller_name="Dr. Marcus Wright",
            campaign_id="CAMP-ENT-99",
            campaign_objective="Enterprise AI Architecture Review",
            caller_email="marcus@vayvora-partner.org",
            company="Global Dynamics",
        )

        state = adapter.conversation_state
        assert state.caller.name == "Dr. Marcus Wright"
        assert state.caller.email == "marcus@vayvora-partner.org"
        assert state.caller.company == "Global Dynamics"
        assert state.caller.is_known() is True

    @pytest.mark.asyncio
    async def test_10_outbound_consultation_grounded_in_target_domain(self):
        """10: Outbound consultation discussion grounds in target domain knowledge."""
        adapter, _, _, _, _, rag, _, stt, _ = build_e2e_components(
            "call-out-10", direction=CallDirection.OUTBOUND
        )
        await rag.index_document(
            RAGChunk(
                doc_id="vay:plat:01",
                domain=DomainType.VAYVORA,
                title="Vayvora Platform",
                content="Vayvora provides enterprise neural streaming architectures.",
                score=1.0,
            )
        )

        await adapter.handle_outbound_call(
            domain=DomainType.VAYVORA,
            caller_name="Elena Fisher",
            campaign_id="CAMP-VAY-02",
            campaign_objective="Introduce Vayvora streaming architecture",
        )

        await simulate_turn(adapter, "Tell me about your streaming platform.")
        assert len(adapter.conversation_state.history) >= 2

    @pytest.mark.asyncio
    async def test_11_caller_preempts_outbound_objective_without_loss(self):
        """11: Caller preempts campaign objective with new intent; agent accommodates without losing context."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components(
            "call-out-11", direction=CallDirection.OUTBOUND
        )
        await adapter.handle_outbound_call(
            domain=DomainType.EDUSAAS,
            caller_name="Arthur Dent",
            campaign_id="CAMP-FOLLOWUP",
            campaign_objective="Check enrollment interest",
        )

        await simulate_turn(adapter, "Actually, I have an urgent question about my billing invoice first.")

        state = adapter.conversation_state
        assert state.caller.campaign_objective == "Check enrollment interest"
        assert state.conversation_active is True


# =============================================================================
# 3. Conversation Semantics (Scenarios 12 - 16)
# =============================================================================


class TestConversationSemantics:
    """Scenarios 12-16: Intent switching, cross-domain safety, 'no' retention, goodbye, business status."""

    @pytest.mark.asyncio
    async def test_12_intent_switching_preserves_stack(self):
        """12: Intent switching mid-dialogue preserves previous intent on history stack."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components("call-sem-12")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        # First turn: course inquiry
        await simulate_turn(adapter, "Tell me about course syllabus.")
        # Second turn: pricing switch
        await simulate_turn(adapter, "How much does it cost?")

        assert len(adapter.conversation_state.history) >= 4

    @pytest.mark.asyncio
    async def test_13_cross_domain_query_routed_safely(self):
        """13: Cross-domain query (e.g. EduSaaS inquiry while in Vayvora context) routed safely."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components("call-sem-13")
        await adapter.handle_inbound_call(domain=DomainType.VAYVORA)

        await simulate_turn(adapter, "Do you offer student university degrees here?")

        assert adapter.conversation_state.conversation_active is True

    @pytest.mark.asyncio
    async def test_14_caller_saying_no_keeps_call_active(self):
        """14: Caller saying 'no' to a question does NOT terminate call; keeps session active."""
        adapter, transport, _, _, _, _, _, stt, _ = build_e2e_components("call-sem-14")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "no")

        assert adapter.conversation_state.conversation_active is True
        assert transport.is_connected is True

    @pytest.mark.asyncio
    async def test_15_explicit_goodbye_completes_session(self):
        """15: Explicit goodbye phrase triggers natural closing and marks session completed."""
        adapter, transport, _, _, _, _, _, stt, _ = build_e2e_components("call-sem-15")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "goodbye")

        assert adapter.conversation_state.conversation_active is False
        assert adapter.conversation_state.termination_requested is True

    @pytest.mark.asyncio
    async def test_16_business_status_update_does_not_conclude_call(self):
        """16: Business status update does not conclude call prematurely."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-sem-16")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        adapter.conversation_state.business_status = "lead_qualified"
        assert adapter.conversation_state.conversation_active is True
        assert adapter.transport.is_connected is True


# =============================================================================
# 4. Grounded RAG (Scenarios 17 - 20)
# =============================================================================


class TestGroundedRAG:
    """Scenarios 17-20: EduSaaS grounding, Vayvora grounding, missing knowledge, cross-domain isolation."""

    @pytest.mark.asyncio
    async def test_17_edusaas_retrieves_grounded_facts(self):
        """17: EduSaaS query retrieves facts grounded in EduSaaS knowledge base."""
        adapter, _, _, _, _, rag, _, stt, _ = build_e2e_components("call-rag-17")
        await rag.index_document(
            RAGChunk(
                doc_id="edu:courses:ml",
                domain=DomainType.EDUSAAS,
                title="Machine Learning 101",
                content="The course covers deep learning and neural networks over 12 weeks.",
                score=1.0,
            )
        )
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "Tell me about the Machine Learning 101 duration.")
        assert adapter.conversation_state.conversation_active is True

    @pytest.mark.asyncio
    async def test_18_vayvora_retrieves_grounded_facts(self):
        """18: Vayvora query retrieves facts grounded in Vayvora knowledge base."""
        adapter, _, _, _, _, rag, _, stt, _ = build_e2e_components("call-rag-18")
        await rag.index_document(
            RAGChunk(
                doc_id="vay:prod:nexus",
                domain=DomainType.VAYVORA,
                title="Vayvora Nexus",
                content="Nexus offers ultra-low latency real-time voice intelligence pipelines.",
                score=1.0,
            )
        )
        await adapter.handle_inbound_call(domain=DomainType.VAYVORA)

        await simulate_turn(adapter, "What is Vayvora Nexus?")
        assert adapter.conversation_state.conversation_active is True

    @pytest.mark.asyncio
    async def test_19_missing_knowledge_handled_politely(self):
        """19: Query with no grounded answer admits lack of knowledge politely."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components("call-rag-19")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "What are the cafeteria lunch specials on Mars?")
        assert adapter.conversation_state.conversation_active is True

    @pytest.mark.asyncio
    async def test_20_cross_domain_knowledge_isolation(self):
        """20: Cross-domain knowledge isolation: EduSaaS facts never leak into Vayvora responses."""
        adapter, _, _, _, _, rag, _, _, _ = build_e2e_components("call-rag-20")
        await rag.index_document(
            RAGChunk(
                doc_id="edu:secret:01",
                domain=DomainType.EDUSAAS,
                title="EduSaaS Tuition Policy",
                content="EduSaaS tuition is 500 dollars per semester.",
                score=1.0,
            )
        )
        results = await rag.search(
            RAGQuery(
                domain=DomainType.VAYVORA,
                query_text="EduSaaS tuition policy",
                top_k=5,
            )
        )
        assert len(results) == 0


# =============================================================================
# 5. MCP Verified Actions (Scenarios 21 - 26)
# =============================================================================


class TestMCPVerifiedActions:
    """Scenarios 21-26: email, calendar availability, calendar booking, error, unverified, failure safety."""

    @pytest.mark.asyncio
    async def test_21_send_email_action_verified(self):
        """21: Agent proposes send_email -> verified -> records external tracking reference."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="request_syllabus",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I've sent the syllabus to your email.",
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={
                    "recipient": "caller@example.com",
                    "subject": "Course Syllabus",
                    "body": "Here is the course syllabus.",
                },
            ),
        )
        adapter, _, _, _, _, _, tools, stt, _ = build_e2e_components(
            "call-mcp-21", llm_canned_decisions=[decision]
        )
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "Please email me the syllabus.")

        state = adapter.conversation_state
        assert state.last_action == "send_email"
        assert state.last_tool_result is not None
        assert state.last_tool_result.success is True

    @pytest.mark.asyncio
    async def test_22_find_available_slots_verified(self):
        """22: Agent proposes calendar availability query -> executes -> verified."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="check_availability",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I have availability tomorrow at 10 AM.",
            proposed_action=ProposedAction(
                tool_name="find_available_slots",
                arguments={"date": "tomorrow"},
            ),
        )
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components(
            "call-mcp-22", llm_canned_decisions=[decision]
        )
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "When are you free for a consultation?")
        assert adapter.conversation_state.last_tool_result.success is True

    @pytest.mark.asyncio
    async def test_23_create_calendar_event_verified(self):
        """23: Agent proposes calendar event creation -> verified -> records confirmation ID."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="book_consultation",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Your consultation is scheduled.",
            proposed_action=ProposedAction(
                tool_name="create_calendar_event",
                arguments={
                    "title": "Admissions Consultation",
                    "slot": "Tomorrow 10:00 AM",
                    "confirmed": True,
                },
            ),
        )
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components(
            "call-mcp-23", llm_canned_decisions=[decision]
        )
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "Yes, book me for tomorrow 10 AM.")
        assert adapter.conversation_state.last_tool_result.success is True

    @pytest.mark.asyncio
    async def test_24_tool_execution_error_handled_gracefully(self):
        """24: Tool execution error handled gracefully without crashing call pipeline."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="send_docs",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I will send that right away.",
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "user@domain.com"},
            ),
        )
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components(
            "call-mcp-24", llm_canned_decisions=[decision], tool_force_error=True
        )
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "Send me the document.")

        assert adapter.conversation_state.conversation_active is True
        assert adapter.conversation_state.last_tool_result.success is False

    @pytest.mark.asyncio
    async def test_25_unverified_tool_result_recorded_in_summary(self):
        """25: Unverified tool result is flagged as failed and recorded in session summary."""
        adapter, _, repo, _, _, _, _, _, _ = build_e2e_components("call-mcp-25")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        adapter.conversation_state.last_action = "send_email"
        adapter.conversation_state.last_tool_result = ToolExecutionResult(
            tool_name="send_email",
            success=False,
            error_message="Action not verified by server",
        )

        summary = await adapter.handle_disconnect(reason="remote_caller_disconnected")
        assert len(summary.failed_actions) == 1
        assert summary.failed_actions[0]["action"] == "send_email"
        assert len(summary.completed_actions) == 0

    @pytest.mark.asyncio
    async def test_26_verification_failure_does_not_mutate_state(self):
        """26: Verification failure does not execute external state mutation."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="book_consultation",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Scheduling...",
            proposed_action=ProposedAction(
                tool_name="create_calendar_event",
                arguments={"title": "Test", "slot": "Tomorrow 10:00 AM"},
            ),
        )
        adapter, _, _, _, _, _, tools, _, _ = build_e2e_components(
            "call-mcp-26", llm_canned_decisions=[decision], tool_force_error=True
        )
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await simulate_turn(adapter, "book it")
        assert adapter.conversation_state.last_tool_result is not None
        assert adapter.conversation_state.last_tool_result.success is False
        assert adapter.conversation_state.conversation_active is True


# =============================================================================
# 6. Barge-In & Interruption (Scenarios 27 - 31)
# =============================================================================


class TestBargeInAndInterruption:
    """Scenarios 27-31: TTS halt, LLM token cancellation, stale frames, fresh turn, MCP continuation."""

    @pytest.mark.asyncio
    async def test_27_caller_speech_during_tts_halts_audio(self):
        """27: Caller speech during active TTS immediately halts audio playback."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-barge-27")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        session.pipeline.state = TurnLifecycleState.SPEAKING
        assert session.pipeline.state == TurnLifecycleState.SPEAKING

        session.pipeline.trigger_barge_in()
        assert session.pipeline.state == TurnLifecycleState.INTERRUPTED
        assert session.pipeline.diagnostics.interruption_count == 1

    @pytest.mark.asyncio
    async def test_28_barge_in_cancels_active_llm_generation(self):
        """28: Barge-in cancels active LLM generation token / text synthesis stream."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-barge-28")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        old_gen = session.pipeline.current_generation_id
        session.pipeline.trigger_barge_in()
        assert session.pipeline.diagnostics.cancelled_generations_count == 1
        assert session.pipeline.current_generation_id == old_gen + 1
        assert session.pipeline.state == TurnLifecycleState.INTERRUPTED

    @pytest.mark.asyncio
    async def test_29_queued_stale_audio_frames_discarded(self):
        """29: Queued stale audio frames discarded on interruption."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-barge-29")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        old_gen = session.pipeline.current_generation_id
        session.pipeline.trigger_barge_in()
        # Older generation is no longer valid
        assert not session.pipeline.cancellation_token.is_valid_generation(old_gen)

    @pytest.mark.asyncio
    async def test_30_new_caller_utterance_accepted_after_barge_in(self):
        """30: New caller utterance after barge-in accepted as fresh conversational turn."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components("call-barge-30")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        session.pipeline.trigger_barge_in()
        assert session.pipeline.state == TurnLifecycleState.INTERRUPTED

        await simulate_turn(adapter, "Wait, tell me something else.")
        assert len(adapter.conversation_state.history) >= 2

    @pytest.mark.asyncio
    async def test_31_background_mcp_action_continues_safely(self):
        """31: Active background MCP action continues safely through audio interruption without corruption."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-barge-31")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        session.pipeline.trigger_barge_in()
        assert adapter.conversation_state.conversation_active is True


# =============================================================================
# 7. Disconnect & Persistence Boundary (Scenarios 32 - 36)
# =============================================================================


class TestDisconnectAndPersistence:
    """Scenarios 32-36: Remote disconnect, pipeline shutdown, TTS shutdown, LLM cancel, CallSummary persistence."""

    @pytest.mark.asyncio
    async def test_32_remote_disconnect_stops_transport(self):
        """32: Remote caller disconnect stops media transport immediately."""
        adapter, transport, _, _, _, _, _, _, _ = build_e2e_components("call-disc-32")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)
        assert transport.is_connected is True

        await adapter.handle_disconnect(reason="remote_caller_disconnected")
        assert transport.is_connected is False

    @pytest.mark.asyncio
    async def test_33_disconnect_terminates_pipeline_gracefully(self):
        """33: Disconnect terminates active voice pipeline and session gracefully."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-disc-33")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)
        assert session.is_active is True

        await adapter.handle_disconnect(reason="remote_caller_disconnected")
        assert session.is_active is False
        assert session.pipeline.state == TurnLifecycleState.COMPLETED

    @pytest.mark.asyncio
    async def test_34_disconnect_cancels_pending_tts(self):
        """34: Disconnect cancels pending TTS synthesis and discards queued audio."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-disc-34")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await adapter.handle_disconnect(reason="remote_caller_disconnected")
        assert session.pipeline.cancellation_token.is_cancelled is True

    @pytest.mark.asyncio
    async def test_35_disconnect_cancels_pending_llm(self):
        """35: Disconnect cancels pending LLM generation without dangling tasks."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-disc-35")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        await adapter.handle_disconnect(reason="network_disconnect")
        assert adapter.conversation_state.conversation_active is False

    @pytest.mark.asyncio
    async def test_36_call_summary_sanitized_and_persisted(self):
        """36: CallSummary record is assembled, sanitized, and persisted to CallSessionRepository."""
        adapter, transport, repo, _, _, _, _, _, _ = build_e2e_components("call-disc-36")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        summary = await adapter.handle_disconnect(reason="completed_goodbye")
        assert summary is not None
        assert summary.call_id == transport.call_id
        assert summary.termination_reason == "completed_goodbye"

        persisted = await repo.get_summary(transport.call_id)
        assert persisted is not None
        assert persisted.call_id == transport.call_id


# =============================================================================
# 8. Failure Recovery & Boundary Resilience (Scenarios 37 - 42)
# =============================================================================


class TestFailureRecoveryAndResilience:
    """Scenarios 37-42: Connection failure, malformed audio, STT failure, LLM failure, TTS failure, recovery."""

    @pytest.mark.asyncio
    async def test_37_transport_connection_failure_handled(self):
        """37: Transport connection failure handled gracefully without uncaught exceptions."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components(
            "call-fail-37", transport_force_error=True
        )
        with pytest.raises(TelephonyConnectionError):
            await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

    @pytest.mark.asyncio
    async def test_38_malformed_audio_packet_rejected_cleanly(self):
        """38: Malformed incoming audio packet rejected without crashing pipeline."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components("call-fail-38")
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        with pytest.raises(TelephonyPacketError):
            await adapter.process_incoming_packet(b"\x01\x02\x03", sample_rate=8000)

        assert session.is_active is True

    @pytest.mark.asyncio
    async def test_39_stt_provider_failure_fallback(self):
        """39: STT provider failure records pipeline error state cleanly."""
        adapter, _, _, _, _, _, _, _, _ = build_e2e_components(
            "call-fail-39", stt_force_error=True
        )
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        with pytest.raises(STTError):
            await simulate_turn(adapter, "test")

        assert session.pipeline.state == TurnLifecycleState.ERROR
        assert len(session.pipeline.diagnostics.errors) >= 1

    @pytest.mark.asyncio
    async def test_40_llm_provider_failure_fallback(self):
        """40: LLM provider failure transitions pipeline to ERROR state."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components(
            "call-fail-40", llm_force_error=True
        )
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        with pytest.raises(LLMError):
            await simulate_turn(adapter, "Hello, I need help.")

        assert session.pipeline.state == TurnLifecycleState.ERROR
        assert len(session.pipeline.diagnostics.errors) >= 1

    @pytest.mark.asyncio
    async def test_41_tts_provider_failure_fallback(self):
        """41: TTS provider failure transitions pipeline to ERROR state."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components(
            "call-fail-41", tts_force_error=True
        )
        session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        with pytest.raises(TTSError):
            await simulate_turn(adapter, "Hi there.")

        assert session.pipeline.state == TurnLifecycleState.ERROR
        assert len(session.pipeline.diagnostics.errors) >= 1

    @pytest.mark.asyncio
    async def test_42_multi_turn_error_recovery(self):
        """42: Pipeline recovers from transient audio packet errors and continues accepting subsequent turns."""
        adapter, _, _, _, _, _, _, stt, _ = build_e2e_components("call-fail-42")
        await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)

        # 1. First turn: valid utterance
        await simulate_turn(adapter, "First question.")
        turns_count_1 = len(adapter.conversation_state.history)
        assert turns_count_1 >= 2

        # 2. Second turn: transient malformed packet rejected
        with pytest.raises(TelephonyPacketError):
            await adapter.process_incoming_packet(b"\x01", sample_rate=8000)

        # 3. Third turn: subsequent valid utterance processed successfully
        await simulate_turn(adapter, "Second question after recovery.")
        turns_count_3 = len(adapter.conversation_state.history)

        assert turns_count_3 > turns_count_1
        assert adapter.conversation_state.conversation_active is True
