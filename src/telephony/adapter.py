"""Telephony-Voice Application Adapter coordinating media streaming and lifecycle.

Connects external TelephonyTransport to VoiceSession and ConversationState,
validating audio specifications, coordinating barge-in, handling remote disconnects,
and persisting finalized CallSummary records.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from src.audio.schemas import AudioChunk, STTResult, TTSResult, VADResult
from src.core.decision import ConversationalDecision
from src.core.errors import TelephonyConnectionError, TelephonyError
from src.core.types import CallDirection, DomainType
from src.logging import get_logger
from src.state.manager import ConversationStateManager
from src.state.models import ConversationState
from src.telephony.audio import TelephonyAudioConverter
from src.telephony.interfaces import CallSessionRepository, TelephonyTransport
from src.telephony.models import (
    CallObservabilityDiagnostics,
    CallSummary,
    TelephonyEvent,
    TelephonyEventType,
)
from src.voice.adapters import (
    PipecatConversationAdapter,
    PipecatSTTAdapter,
    PipecatTTSAdapter,
    PipecatVADAdapter,
)
from src.voice.context import TurnLifecycleState
from src.voice.events import (
    CallerInterrupted,
    SpeechStarted,
    TranscriptFinal,
    TTSStarted,
    VoiceEvent,
)
from src.voice.pipeline import VoicePipeline
from src.voice.session import MockAudioInput, MockAudioOutput, VoiceSession

logger = get_logger("telephony.adapter")


class TelephonyAudioOutputSink:
    """Audio output sink transmitting synthesized TTS frames to TelephonyTransport."""

    def __init__(self, transport: TelephonyTransport, converter: TelephonyAudioConverter) -> None:
        self.transport = transport
        self.converter = converter
        self.sent_frames: List[bytes] = []

    def write(self, data: bytes) -> None:
        """Forward synthesized WAV/PCM audio to the external transport."""
        # Convert to raw PCM if WAV header present
        pcm_bytes = data[44:] if data.startswith(b"RIFF") and len(data) > 44 else data
        self.sent_frames.append(pcm_bytes)
        # Note: transport send is async, recorded for delivery

    def clear(self) -> None:
        """Purge pending frames on barge-in."""
        self.sent_frames.clear()
        logger.info("TelephonyAudioOutputSink: queued frames purged on barge-in.")


class TelephonyVoiceAdapter:
    """Application-side coordinator mediating between TelephonyTransport and VoiceSession."""

    def __init__(
        self,
        transport: TelephonyTransport,
        state_manager: ConversationStateManager,
        vad_provider: Any,
        stt_provider: Any,
        conversation_engine: Any,
        tts_provider: Any,
        repository: Optional[CallSessionRepository] = None,
        audio_converter: Optional[TelephonyAudioConverter] = None,
    ) -> None:
        self.transport = transport
        self.state_manager = state_manager
        self.vad_provider = vad_provider
        self.stt_provider = stt_provider
        self.engine = conversation_engine
        self.tts_provider = tts_provider
        self.repository = repository
        self.audio_converter = audio_converter or TelephonyAudioConverter()

        self.voice_session: Optional[VoiceSession] = None
        self.conversation_state: Optional[ConversationState] = None
        self.observability = CallObservabilityDiagnostics(
            call_id=transport.call_id,
            session_id=transport.session_id,
        )

        self._start_time: float = 0.0
        self._end_time: Optional[float] = None
        self._output_sink: Optional[TelephonyAudioOutputSink] = None

    # -------------------------------------------------------------------------
    # Inbound Call Lifecycle
    # -------------------------------------------------------------------------

    async def handle_inbound_call(
        self,
        domain: DomainType = DomainType.EDUSAAS,
        caller_name: Optional[str] = None,
    ) -> VoiceSession:
        """Establish inbound call session from external telephony media transport."""
        # 1. Connect transport
        await self.transport.connect()
        self._start_time = time.time()
        self.observability.connection_time = self._start_time

        await self.transport.send_event(
            TelephonyEvent(
                event_type=TelephonyEventType.CALL_CONNECTED,
                call_id=self.transport.call_id,
                payload={"direction": "inbound", "caller": self.transport.caller_number},
            )
        )

        # 2. Create inbound ConversationState (supports unknown caller)
        state = self.state_manager.create_inbound_state(
            call_id=self.transport.call_id,
            caller_phone=self.transport.caller_number or "+15551234567",
            domain=domain,
            caller_name=caller_name,
        )
        self.conversation_state = state

        # 3. Assemble VoicePipeline and VoiceSession
        session = self._build_voice_session(state=state)
        session.start()
        self.voice_session = session

        await self.transport.send_event(
            TelephonyEvent(
                event_type=TelephonyEventType.MEDIA_STARTED,
                call_id=self.transport.call_id,
                payload={"sample_rate": self.audio_converter.target_sample_rate},
            )
        )

        logger.info(
            "Telephony inbound call %s initialized for %s",
            self.transport.call_id,
            domain.value,
        )
        return session

    # -------------------------------------------------------------------------
    # Outbound Call Lifecycle
    # -------------------------------------------------------------------------

    async def handle_outbound_call(
        self,
        domain: DomainType,
        caller_name: str,
        campaign_id: str,
        campaign_objective: str,
        caller_email: Optional[str] = None,
        company: Optional[str] = None,
        known_purpose: Optional[str] = None,
    ) -> VoiceSession:
        """Establish outbound call session with pre-populated campaign and caller context."""
        await self.transport.connect()
        self._start_time = time.time()
        self.observability.connection_time = self._start_time

        await self.transport.send_event(
            TelephonyEvent(
                event_type=TelephonyEventType.CALL_CONNECTED,
                call_id=self.transport.call_id,
                payload={"direction": "outbound", "callee": self.transport.callee_number},
            )
        )

        # Create enriched outbound ConversationState
        state = self.state_manager.create_outbound_state(
            call_id=self.transport.call_id,
            caller_phone=self.transport.callee_number or "+15551234567",
            domain=domain,
            caller_name=caller_name,
            campaign_id=campaign_id,
            campaign_objective=campaign_objective,
            caller_email=caller_email,
            company=company,
            known_purpose=known_purpose,
        )
        self.conversation_state = state

        session = self._build_voice_session(state=state)
        await session.start_outbound(self.engine)
        self.voice_session = session

        await self.transport.send_event(
            TelephonyEvent(
                event_type=TelephonyEventType.MEDIA_STARTED,
                call_id=self.transport.call_id,
                payload={"campaign_id": campaign_id},
            )
        )

        logger.info(
            "Telephony outbound call %s initialized for %s (%s)",
            self.transport.call_id,
            domain.value,
            campaign_id,
        )
        return session

    # -------------------------------------------------------------------------
    # Media Packet Ingestion
    # -------------------------------------------------------------------------

    async def process_incoming_packet(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        timestamp: float = 0.0,
        encoding: str = "pcm16",
    ) -> Tuple[VADResult, Optional[STTResult], Optional[TTSResult]]:
        """Validate and ingest incoming telephony packet, returning turn outcomes."""
        if not self.voice_session or not self.voice_session.is_active:
            raise TelephonyError("Cannot process media: call session is not active.")

        if self.observability.first_audio_time is None:
            self.observability.first_audio_time = time.time()

        # 1. Validate & Convert to AudioChunk
        chunk = self.audio_converter.telephony_frame_to_chunk(
            raw_bytes=audio_bytes,
            sample_rate=sample_rate,
            channels=channels,
            timestamp=timestamp,
            encoding=encoding,
        )

        # 2. Process through VoiceSession
        vad_res, stt_res, tts_res = await self.voice_session.process_audio_chunk(chunk)

        # 3. Transmit synthesized response through transport if turn completed
        if tts_res and tts_res.audio_data and self.transport.is_connected:
            # Deliver to external telephony transport
            pcm_payload = self.audio_converter.chunk_to_telephony_frame(
                AudioChunk(
                    data=tts_res.audio_data,
                    sample_rate=tts_res.sample_rate,
                    channels=1,
                    format="wav",
                )
            )
            await self.transport.send_audio(pcm_payload)

        # 4. Check if conversation completed naturally
        if (
            self.conversation_state
            and not self.conversation_state.conversation_active
            and self.transport.is_connected
        ):
            await self.handle_disconnect(reason=self.conversation_state.termination_reason or "completed")

        return vad_res, stt_res, tts_res

    # -------------------------------------------------------------------------
    # Disconnect & Session Persistence
    # -------------------------------------------------------------------------

    async def handle_disconnect(
        self,
        reason: str = "remote_caller_disconnected",
    ) -> Optional[CallSummary]:
        """Gracefully terminate media pipeline, stop session, and persist CallSummary."""
        if self._end_time is not None:
            # Already disconnected
            return None

        self._end_time = time.time()
        self.observability.disconnect_time = self._end_time
        logger.info("Call %s disconnected. Reason: %s", self.transport.call_id, reason)

        # 1. Send signaling events
        if self.transport.is_connected:
            await self.transport.send_event(
                TelephonyEvent(
                    event_type=TelephonyEventType.MEDIA_STOPPED,
                    call_id=self.transport.call_id,
                )
            )
            await self.transport.send_event(
                TelephonyEvent(
                    event_type=TelephonyEventType.CALL_DISCONNECTED,
                    call_id=self.transport.call_id,
                    payload={"reason": reason},
                )
            )

        # 2. Stop voice session and cancel active generation
        if self.voice_session:
            self.voice_session.stop(reason=reason)

        # 3. Disconnect transport
        if self.transport.is_connected:
            await self.transport.disconnect()

        # 4. Assemble CallSummary
        summary: Optional[CallSummary] = None
        if self.conversation_state:
            summary = CallSummary.from_conversation_state(
                state=self.conversation_state,
                session_id=self.transport.session_id,
                start_time=self._start_time,
                end_time=self._end_time,
                disconnect_reason=reason,
            )

            # 5. Persist to repository
            if self.repository:
                await self.repository.persist_summary(summary)

        return summary

    # -------------------------------------------------------------------------
    # Voice Session Builder
    # -------------------------------------------------------------------------

    def _build_voice_session(self, state: ConversationState) -> VoiceSession:
        """Construct decoupled VoicePipeline, adapters, and session."""
        vad_adapter = PipecatVADAdapter(self.vad_provider)
        stt_adapter = PipecatSTTAdapter(self.stt_provider)
        conversation_adapter = PipecatConversationAdapter(self.engine)
        tts_adapter = PipecatTTSAdapter(self.tts_provider)

        self._output_sink = TelephonyAudioOutputSink(
            transport=self.transport,
            converter=self.audio_converter,
        )

        pipeline = VoicePipeline(
            session_id=self.transport.session_id,
            vad_adapter=vad_adapter,
            stt_adapter=stt_adapter,
            conversation_adapter=conversation_adapter,
            tts_adapter=tts_adapter,
            audio_output=self._output_sink,
        )

        # Hook observability event listener
        def _on_event(event: VoiceEvent) -> None:
            if isinstance(event, TranscriptFinal):
                if self.observability.first_transcript_time is None:
                    self.observability.first_transcript_time = time.time()
                self.observability.turn_id = event.turn_id or 0
            elif isinstance(event, TTSStarted):
                if self.observability.first_tts_audio_time is None:
                    self.observability.first_tts_audio_time = time.time()
            elif isinstance(event, CallerInterrupted):
                self.observability.interruptions += 1
                if self._output_sink:
                    self._output_sink.clear()

        pipeline.subscribe(_on_event)

        return VoiceSession(
            session_id=self.transport.session_id,
            conversation_state=state,
            pipeline=pipeline,
            audio_input=MockAudioInput(),
            audio_output=self._output_sink,
        )
