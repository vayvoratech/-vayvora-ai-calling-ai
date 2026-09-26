"""Workbench service layer orchestrating session lifecycle, turn processing, and debug diagnostics.

Decouples the Streamlit presentation tier from the underlying engine, state manager,
RAG pipeline, and MCP tool layer.
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypeVar

from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.interfaces import KnowledgeProvider, LLMProvider, ToolProvider
from src.core.types import (
    CallDirection,
    CallMetadata,
    CallStatus,
    ConversationStage,
    DomainType,
    ToolExecutionResult,
)
from src.logging import get_logger
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState

logger = get_logger("ui.service")

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Execute an asynchronous coroutine synchronously in any threading context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(coro)).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class RuntimeMode(str, Enum):
    """Runtime execution mode for workbench services."""

    MOCK = "MOCK"
    LIVE = "LIVE"
    HYBRID = "HYBRID"


@dataclass
class ServiceComponents:
    """Container for instantiated voice agent subsystems."""

    engine: ConversationEngine
    state_manager: ConversationStateManager
    llm_provider: LLMProvider
    knowledge_provider: Optional[KnowledgeProvider]
    tool_provider: Optional[ToolProvider]
    vad_provider: Optional[Any] = None
    stt_provider: Optional[Any] = None
    tts_provider: Optional[Any] = None
    llm_mode: str = "MOCK"
    rag_mode: str = "MOCK"
    tool_mode: str = "MOCK"
    vad_mode: str = "MOCK"
    stt_mode: str = "MOCK"
    tts_mode: str = "MOCK"
    overall_mode: RuntimeMode = RuntimeMode.MOCK
    initialization_notes: List[str] = field(default_factory=list)


class WorkbenchService:
    """Application facade mediating between Streamlit and the core ConversationEngine."""

    def __init__(self, components: ServiceComponents) -> None:
        self.engine = components.engine
        self.state_manager = components.state_manager
        self.llm_provider = components.llm_provider
        self.knowledge_provider = components.knowledge_provider
        self.tool_provider = components.tool_provider
        self.vad_provider = components.vad_provider
        self.stt_provider = components.stt_provider
        self.tts_provider = components.tts_provider
        self.llm_mode = components.llm_mode
        self.rag_mode = components.rag_mode
        self.tool_mode = components.tool_mode
        self.vad_mode = components.vad_mode
        self.stt_mode = components.stt_mode
        self.tts_mode = components.tts_mode
        self.overall_mode = components.overall_mode
        self.initialization_notes = components.initialization_notes
        self.last_error: Optional[str] = None
        self.last_turn_result: Optional[EngineTurnResult] = None
        self.active_voice_session: Optional[Any] = None
        self.active_telephony_adapter: Optional[Any] = None
        from src.telephony.repository import MockCallSessionRepository
        self.call_repository = MockCallSessionRepository()



    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def create_inbound_session(
        self,
        call_id: str,
        caller_phone: str,
        domain: DomainType = DomainType.EDUSAAS,
        caller_name: Optional[str] = None,
        caller_email: Optional[str] = None,
        caller_company: Optional[str] = None,
    ) -> ConversationState:
        """Create a fresh inbound session in ConversationStateManager."""
        self.last_error = None
        self.last_turn_result = None
        return self.state_manager.create_inbound_state(
            call_id=call_id,
            caller_phone=caller_phone,
            domain=domain,
            caller_name=caller_name,
            caller_email=caller_email,
            caller_company=caller_company,
        )

    def create_outbound_session(
        self,
        call_id: str,
        caller_phone: str,
        domain: DomainType,
        caller_name: str,
        campaign_id: str,
        campaign_objective: str,
        caller_email: Optional[str] = None,
        company: Optional[str] = None,
        known_purpose: Optional[str] = None,
    ) -> ConversationState:
        """Create a fresh enriched outbound session in ConversationStateManager."""
        self.last_error = None
        state = self.state_manager.create_outbound_state(
            call_id=call_id,
            caller_phone=caller_phone,
            domain=domain,
            caller_name=caller_name,
            campaign_id=campaign_id,
            campaign_objective=campaign_objective,
            caller_email=caller_email,
            company=company,
            known_purpose=known_purpose,
        )
        try:
            opening_turn = run_sync(self.engine.start_outbound_conversation(state))
            self.last_turn_result = opening_turn
            self.state_manager.save(state)
        except Exception as exc:
            logger.warning("Could not automatically initiate outbound opening: %s", exc)
        return state

    def get_session(self, session_id: str) -> Optional[ConversationState]:
        """Fetch active session from memory."""
        return self.state_manager.get(session_id)

    def reset_session(self, session_id: str) -> None:
        """Completely delete and unregister a conversation session."""
        self.last_error = None
        self.last_turn_result = None
        self.state_manager.delete(session_id)
        logger.info("Session %s reset and removed from manager.", session_id)

    # -------------------------------------------------------------------------
    # Turn Execution
    # -------------------------------------------------------------------------

    async def process_turn_async(
        self, session_id: str, user_message: str
    ) -> Tuple[ConversationState, EngineTurnResult]:
        """Asynchronously process a user message through the ConversationEngine."""
        state = self.get_session(session_id)
        if not state:
            raise KeyError(f"Session '{session_id}' not found in state manager.")

        try:
            self.last_error = None
            turn_result = await self.engine.process_user_turn(state, user_message)
            self.last_turn_result = turn_result
            # Update state in storage
            self.state_manager.save(state)
            return state, turn_result
        except Exception as exc:
            self.last_error = f"Engine turn processing error: {exc}"
            logger.error("Turn failed for session %s: %s", session_id, exc)
            # Create a safe fallback turn result to prevent UI crash
            fallback_turn = EngineTurnResult(
                response_text=(
                    "I encountered an unexpected internal error processing that request. "
                    "Our system has logged the details."
                ),
                decision=ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="system_error",
                    proposed_stage=state.stage,
                    user_facing_response="Internal error",
                ),
            )
            self.last_turn_result = fallback_turn
            return state, fallback_turn

    def process_turn(
        self, session_id: str, user_message: str
    ) -> Tuple[ConversationState, EngineTurnResult]:
        """Synchronously process a user message (suitable for Streamlit execution)."""
        return run_sync(self.process_turn_async(session_id, user_message))

    # -------------------------------------------------------------------------
    # Debug Diagnostics
    # -------------------------------------------------------------------------

    def get_debug_payload(
        self,
        state: Optional[ConversationState],
        turn_result: Optional[EngineTurnResult] = None,
    ) -> Dict[str, Any]:
        """Construct structured, sanitized developer diagnostics."""
        result = turn_result or self.last_turn_result

        if not state:
            return {
                "session": {"status": "No active session"},
                "runtime": {
                    "overall_mode": self.overall_mode.value,
                    "llm_mode": self.llm_mode,
                    "rag_mode": self.rag_mode,
                    "tool_mode": self.tool_mode,
                },
                "errors": {"last_error": self.last_error},
            }

        # 1. Session Metrics
        session_info = {
            "session_id": state.metadata.call_id,
            "company": state.current_domain.value,
            "direction": state.metadata.direction.value,
            "conversation_active": state.conversation_active,
            "conversation_stage": state.stage.value,
            "business_status": state.business_status,
            "termination_state": {
                "termination_requested": state.termination_requested,
                "termination_reason": state.termination_reason,
            },
        }

        # 2. Intent Metrics
        intent_info = {
            "current_intent": state.current_intent,
            "sub_intent": state.current_sub_intent,
            "intent_history": list(state.intent_history),
        }

        # 3. Caller Profile (Sanitized)
        is_known_val = state.caller.is_known() if callable(getattr(state.caller, "is_known", None)) else bool(state.caller.is_known)
        caller_info = {
            "name": state.caller.name,
            "email": state.caller.email,
            "phone": state.caller.phone,
            "company": state.caller.company,
            "is_known": is_known_val,
        }

        # 4. Extracted Entities
        entities_info = {
            "extracted_slots": dict(state.extracted_slots),
            "current_interest": state.current_interest,
            "pending_question": state.pending_question,
        }

        # 5. RAG Diagnostics
        rag_info: Dict[str, Any] = {
            "knowledge_required": False,
            "knowledge_query": None,
            "knowledge_available": False,
            "retrieved_sources": [],
            "grounded_citations": [],
            "domain": state.current_domain.value if state else "general",
            "hybrid_pipeline": "Dense + BM25 + RRF + Reranker",
        }
        if result and result.decision:
            rag_info["knowledge_required"] = bool(result.decision.knowledge_required)
            rag_info["knowledge_query"] = result.decision.knowledge_query
            rag_info["knowledge_available"] = bool(result.grounded_citations)
            rag_info["retrieved_sources"] = list(result.grounded_citations)
            rag_info["grounded_citations"] = list(result.grounded_citations)

        # 6. Tools Diagnostics
        tool_status = "none"
        verification_status = "none"
        external_reference = None
        failure_reason = None

        if state.last_tool_result:
            tool_status = "succeeded" if state.last_tool_result.success else "failed"
            verification_status = "verified" if state.last_tool_result.success else "unverified"
            external_reference = state.last_tool_result.verification_code
            failure_reason = state.last_tool_result.error_message
        elif state.pending_action:
            tool_status = "pending"
            verification_status = "pending"

        proposed_action_dict = None
        if result and result.decision and result.decision.action_proposed and result.decision.proposed_action:
            proposed_action_dict = {
                "tool_name": result.decision.proposed_action.tool_name,
                "arguments": result.decision.proposed_action.arguments,
            }

        tools_info = {
            "proposed_action": proposed_action_dict,
            "last_action": state.last_action,
            "tool_status": tool_status,
            "verification_status": verification_status,
            "external_reference": external_reference,
            "failure_reason": failure_reason,
        }

        # 7. Latency Metrics
        latency_info = {
            "total_turn_ms": result.latency_ms if result else None,
            "breakdown": result.latency_breakdown if result else {},
        }

        # 8. Runtime and Errors
        return {
            "session": session_info,
            "intent": intent_info,
            "caller": caller_info,
            "entities": entities_info,
            "rag": rag_info,
            "tools": tools_info,
            "latency": latency_info,
            "runtime": {
                "overall_mode": self.overall_mode.value,
                "llm_mode": self.llm_mode,
                "rag_mode": self.rag_mode,
                "tool_mode": self.tool_mode,
                "vad_mode": self.vad_mode,
                "stt_mode": self.stt_mode,
                "tts_mode": self.tts_mode,
                "initialization_notes": self.initialization_notes,
            },
            "errors": {
                "last_error": self.last_error,
            },
        }

    # -------------------------------------------------------------------------
    # Audio Subsystems Testing Facade
    # -------------------------------------------------------------------------

    def test_vad(self, audio_data: bytes, sample_rate: int = 16000) -> Any:
        """Evaluate raw PCM audio chunk through the active VAD provider."""
        if not self.vad_provider:
            raise RuntimeError("No VAD provider configured in WorkbenchService.")
        from src.audio.schemas import AudioChunk

        chunk = AudioChunk(
            data=audio_data,
            sample_rate=sample_rate,
            channels=1,
            format="pcm16",
        )
        return self.vad_provider.process_chunk(chunk)

    def test_stt(self, audio_data: Any, sample_rate: int = 16000) -> Any:
        """Transcribe speech audio bytes or segment through the active STT provider."""
        if not self.stt_provider:
            raise RuntimeError("No STT provider configured in WorkbenchService.")
        return run_sync(self.stt_provider.transcribe(audio_data, sample_rate=sample_rate))

    def test_tts(self, text: str, voice_id: Optional[str] = None) -> Any:
        """Synthesize text into speech audio through the active TTS provider."""
        if not self.tts_provider:
            raise RuntimeError("No TTS provider configured in WorkbenchService.")
        return run_sync(self.tts_provider.synthesize(text, voice_id=voice_id))

    # -------------------------------------------------------------------------
    # Phase 8: Real-Time Voice Session Orchestration
    # -------------------------------------------------------------------------

    def start_voice_session(self, session_id: str) -> Any:
        """Initialize and start a real-time VoiceSession for the given active call."""
        state = self.get_session(session_id)
        if not state:
            raise RuntimeError(f"Cannot start voice session: call '{session_id}' not found.")

        from src.voice.adapters import (
            PipecatConversationAdapter,
            PipecatSTTAdapter,
            PipecatTTSAdapter,
            PipecatVADAdapter,
        )
        from src.voice.pipeline import VoicePipeline
        from src.voice.session import MockAudioInput, MockAudioOutput, VoiceSession

        vad_adapter = PipecatVADAdapter(self.vad_provider)
        stt_adapter = PipecatSTTAdapter(self.stt_provider)
        tts_adapter = PipecatTTSAdapter(self.tts_provider)
        conversation_adapter = PipecatConversationAdapter(self.engine)
        audio_output = MockAudioOutput()

        pipeline = VoicePipeline(
            session_id=session_id,
            vad_adapter=vad_adapter,
            stt_adapter=stt_adapter,
            conversation_adapter=conversation_adapter,
            tts_adapter=tts_adapter,
            audio_output=audio_output,
        )

        session = VoiceSession(
            session_id=session_id,
            conversation_state=state,
            pipeline=pipeline,
            audio_input=MockAudioInput(),
            audio_output=audio_output,
        )
        session.start()
        self.active_voice_session = session
        return session

    def stop_voice_session(self) -> Optional[Dict[str, Any]]:
        """Gracefully stop the currently active VoiceSession."""
        if not self.active_voice_session:
            return None
        session = self.active_voice_session
        session.stop()
        diag = session.get_diagnostics()
        self.active_voice_session = None
        return diag

    def process_voice_chunk(self, chunk: Any) -> Any:
        """Feed a single audio chunk into the active voice session."""
        if not self.active_voice_session:
            raise RuntimeError("No active VoiceSession found.")
        return run_sync(self.active_voice_session.process_audio_chunk(chunk))

    def simulate_voice_utterance(self, text: str) -> Tuple[Any, Any, Any]:
        """Simulate a complete user speech utterance through VAD, STT, Engine, and TTS."""
        if not self.active_voice_session:
            raise RuntimeError("No active VoiceSession found.")
        from src.audio.schemas import AudioChunk
        from src.audio.tts import create_pcm_wav_bytes

        # If mock STT, queue the exact text to ensure deterministic evaluation
        if hasattr(self.stt_provider, "add_transcription"):
            self.stt_provider.add_transcription(text)

        # 1. Feed speech chunks (active speech frames)
        audio_bytes = create_pcm_wav_bytes(duration=0.3, frequency=440.0)
        speech_chunk = AudioChunk(
            data=audio_bytes,
            sample_rate=16000,
            duration=0.3,
            format="pcm16",
        )
        self.process_voice_chunk(speech_chunk)

        # 2. Feed silence chunk to trigger speech_end and turn completion
        silence_bytes = b"\x00" * 16000  # 0.5s silence
        silence_chunk = AudioChunk(
            data=silence_bytes,
            sample_rate=16000,
            duration=0.5,
            format="pcm16",
        )
        return self.process_voice_chunk(silence_chunk)

    def get_voice_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the current or last voice session."""
        if self.active_voice_session:
            return self.active_voice_session.get_diagnostics()
        return {}

    # -------------------------------------------------------------------------
    # Phase 9: Telephony Simulation Facade
    # -------------------------------------------------------------------------

    def simulate_telephony_inbound(
        self,
        call_id: str,
        caller_phone: str = "+15551234567",
        domain: DomainType = DomainType.EDUSAAS,
    ) -> Any:
        """Simulate an inbound telephony call connection through TelephonyVoiceAdapter."""
        from src.telephony.adapter import TelephonyVoiceAdapter
        from src.telephony.transport import MockTelephonyTransport

        transport = MockTelephonyTransport(
            call_id=call_id,
            session_id=f"voice-session-{call_id}",
            direction=CallDirection.INBOUND,
            caller_number=caller_phone,
        )

        adapter = TelephonyVoiceAdapter(
            transport=transport,
            state_manager=self.state_manager,
            vad_provider=self.vad_provider,
            stt_provider=self.stt_provider,
            conversation_engine=self.engine,
            tts_provider=self.tts_provider,
            repository=self.call_repository,
        )

        session = run_sync(adapter.handle_inbound_call(domain=domain))
        self.active_voice_session = session
        self.active_telephony_adapter = adapter
        return adapter

    def simulate_telephony_outbound(
        self,
        call_id: str,
        caller_name: str,
        caller_phone: str,
        campaign_id: str,
        campaign_objective: str,
        domain: DomainType = DomainType.EDUSAAS,
    ) -> Any:
        """Simulate an outbound telephony call connection through TelephonyVoiceAdapter."""
        from src.telephony.adapter import TelephonyVoiceAdapter
        from src.telephony.transport import MockTelephonyTransport

        transport = MockTelephonyTransport(
            call_id=call_id,
            session_id=f"voice-session-{call_id}",
            direction=CallDirection.OUTBOUND,
            callee_number=caller_phone,
        )

        adapter = TelephonyVoiceAdapter(
            transport=transport,
            state_manager=self.state_manager,
            vad_provider=self.vad_provider,
            stt_provider=self.stt_provider,
            conversation_engine=self.engine,
            tts_provider=self.tts_provider,
            repository=self.call_repository,
        )

        session = run_sync(
            adapter.handle_outbound_call(
                domain=domain,
                caller_name=caller_name,
                campaign_id=campaign_id,
                campaign_objective=campaign_objective,
            )
        )
        self.active_voice_session = session
        self.active_telephony_adapter = adapter
        return adapter

    def simulate_telephony_utterance(self, text: str) -> Tuple[Any, Any, Any]:
        """Simulate an inbound speech utterance arriving as raw telephony media frames."""
        if not self.active_telephony_adapter:
            raise RuntimeError("No active TelephonyVoiceAdapter found.")
        from src.audio.tts import create_pcm_wav_bytes

        if hasattr(self.stt_provider, "add_transcription"):
            self.stt_provider.add_transcription(text)

        # 1. Speech frames
        audio_bytes = create_pcm_wav_bytes(duration=0.3, frequency=440.0)
        pcm_speech = audio_bytes[44:] if audio_bytes.startswith(b"RIFF") else audio_bytes
        run_sync(self.active_telephony_adapter.process_incoming_packet(pcm_speech))

        # 2. Silence frame to trigger turn completion
        pcm_silence = b"\x00" * 16000
        return run_sync(self.active_telephony_adapter.process_incoming_packet(pcm_silence))

    def simulate_telephony_disconnect(
        self, reason: str = "remote_caller_disconnected"
    ) -> Optional[Any]:
        """Simulate remote carrier or caller disconnect, closing session and persisting summary."""
        if not self.active_telephony_adapter:
            return None
        summary = run_sync(self.active_telephony_adapter.handle_disconnect(reason=reason))
        self.active_telephony_adapter = None
        self.active_voice_session = None
        return summary

    def get_telephony_diagnostics(self) -> Dict[str, Any]:
        """Return call observability and transport diagnostics."""
        if self.active_telephony_adapter:
            return {
                "observability": self.active_telephony_adapter.observability.model_dump(),
                "transport": {
                    "call_id": self.active_telephony_adapter.transport.call_id,
                    "direction": self.active_telephony_adapter.transport.direction.value,
                    "is_connected": self.active_telephony_adapter.transport.is_connected,
                    "sent_frames_count": len(self.active_telephony_adapter.transport.sent_audio_frames)
                    if hasattr(self.active_telephony_adapter.transport, "sent_audio_frames")
                    else 0,
                },
            }
        return {}



