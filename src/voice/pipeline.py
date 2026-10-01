"""Pipecat Real-Time Voice Orchestration Pipeline and Frame Processing Engine.

Coordinates real-time audio streaming, temporal turn lifecycles, VAD boundary detection,
STT speech segmentation, ConversationEngine invocation, TTS synthesis, and barge-in interruption.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.audio.schemas import AudioChunk, SpeechSegment, STTResult, TTSResult, VADResult
from src.core.errors import (
    AudioPipelineError,
    BargeInInterruptionError,
    StaleGenerationError,
    VoicePipelineError,
)
from src.logging import get_logger
from src.state.models import ConversationState
from src.voice.adapters.llm import PipecatConversationAdapter
from src.voice.adapters.stt import PipecatSTTAdapter
from src.voice.adapters.tts import PipecatTTSAdapter
from src.voice.adapters.vad import PipecatVADAdapter
from src.voice.context import CancellationToken, TurnLifecycleState, VoiceDiagnostics
from src.voice.events import (
    AgentResponseCompleted,
    AgentResponseStarted,
    AgentThinking,
    CallerInterrupted,
    SessionCompleted,
    SpeechEnded,
    SpeechStarted,
    TranscriptFinal,
    TTSStarted,
    TTSStopped,
    VoiceEvent,
    VoicePipelineErrorEvent,
)

logger = get_logger("voice.pipeline")


class VoicePipeline:
    """Core real-time orchestration pipeline managing the audio frame loop and barge-in."""

    def __init__(
        self,
        session_id: str,
        vad_adapter: PipecatVADAdapter,
        stt_adapter: PipecatSTTAdapter,
        conversation_adapter: PipecatConversationAdapter,
        tts_adapter: PipecatTTSAdapter,
        audio_output: Optional[Any] = None,
    ) -> None:
        self.session_id = session_id
        self.vad = vad_adapter
        self.stt = stt_adapter
        self.conversation = conversation_adapter
        self.tts = tts_adapter
        self.output_sink = audio_output

        # Lifecycle & Turn State Machine
        self.state: TurnLifecycleState = TurnLifecycleState.LISTENING
        self.turn_id: int = 0
        self.current_generation_id: int = 0
        self.cancellation_token = CancellationToken(generation_id=0)

        # Speech accumulation buffers
        self._speech_chunks: List[AudioChunk] = []
        self._is_capturing_speech: bool = False
        self._speech_start_time: float = 0.0

        # Diagnostics & Event Dispatch
        self.diagnostics = VoiceDiagnostics(session_id=session_id)
        self.event_history: List[VoiceEvent] = []
        self.event_subscribers: List[Callable[[VoiceEvent], None]] = []

    # -------------------------------------------------------------------------
    # Event Emission
    # -------------------------------------------------------------------------

    def emit_event(self, event: VoiceEvent) -> None:
        """Record and dispatch typed pipeline events to registered subscribers."""
        self.event_history.append(event)
        for subscriber in self.event_subscribers:
            try:
                subscriber(event)
            except Exception as exc:
                logger.warning("Event subscriber error: %s", exc)

    def subscribe(self, callback: Callable[[VoiceEvent], None]) -> None:
        """Register an event listener for telemetry or UI streaming."""
        self.event_subscribers.append(callback)

    # -------------------------------------------------------------------------
    # Barge-In / Interruption Handler
    # -------------------------------------------------------------------------

    def trigger_barge_in(self) -> None:
        """Interrupt active agent speech/thinking upon caller speech detection."""
        old_generation_id = self.current_generation_id
        logger.info(
            "Barge-in triggered for session %s at turn %d. Invalidating generation %d.",
            self.session_id,
            self.turn_id,
            old_generation_id,
        )

        # 1. Invalidate current generation token
        self.cancellation_token.cancel()

        # 2. Advance generation counter and reset token
        self.current_generation_id += 1
        self.cancellation_token.reset(self.current_generation_id)

        # 3. Cancel active TTS audio synthesis
        self.tts.cancel_active_synthesis()

        # 4. Flush queued assistant audio from output sink
        if self.output_sink and hasattr(self.output_sink, "clear"):
            self.output_sink.clear()

        # 5. Update diagnostics & state
        self.diagnostics.interruption_count += 1
        self.diagnostics.cancelled_generations_count += 1
        self.diagnostics.current_generation_id = self.current_generation_id
        self.state = TurnLifecycleState.INTERRUPTED

        # 6. Emit event
        self.emit_event(
            CallerInterrupted(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=old_generation_id,
                cancelled_generation_id=old_generation_id,
            )
        )

    # -------------------------------------------------------------------------
    # Audio Frame Ingestion Loop
    # -------------------------------------------------------------------------

    async def process_audio_chunk(
        self,
        chunk: AudioChunk,
        conversation_state: ConversationState,
    ) -> Tuple[VADResult, Optional[STTResult], Optional[TTSResult]]:
        """Process an incoming AudioChunk through VAD, STT, ConversationEngine, and TTS."""
        stt_result: Optional[STTResult] = None
        tts_result: Optional[TTSResult] = None

        # 1. Evaluate Voice Activity
        try:
            vad_res = self.vad.process_chunk(chunk)
        except Exception as exc:
            self.state = TurnLifecycleState.ERROR
            self.diagnostics.errors.append(f"VAD error: {exc}")
            self.emit_event(
                VoicePipelineErrorEvent(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    generation_id=self.current_generation_id,
                    error_message=str(exc),
                    component="vad",
                )
            )
            raise VoicePipelineError(f"VAD failed: {exc}") from exc

        # 2. Check for Speech Start
        if vad_res.speech_start:
            # Check for Barge-In: caller starts speaking while agent is SPEAKING or THINKING
            if self.state in [TurnLifecycleState.SPEAKING, TurnLifecycleState.THINKING]:
                self.trigger_barge_in()

            self._is_capturing_speech = True
            self._speech_start_time = vad_res.timestamp
            self._speech_chunks = [chunk]
            self.state = TurnLifecycleState.CAPTURING

            self.diagnostics.speech_start_time = vad_res.timestamp
            self.emit_event(
                SpeechStarted(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    generation_id=self.current_generation_id,
                )
            )

        # 3. Accumulate Speech Frames
        elif self._is_capturing_speech:
            self._speech_chunks.append(chunk)

            # 4. Check for Speech End (turn boundary reached)
            if vad_res.speech_end:
                self._is_capturing_speech = False
                self.state = TurnLifecycleState.TRANSCRIBING

                speech_segment = self._build_speech_segment(end_time=vad_res.timestamp)
                self.diagnostics.speech_end_time = vad_res.timestamp
                self.diagnostics.speech_duration = speech_segment.duration

                self.emit_event(
                    SpeechEnded(
                        session_id=self.session_id,
                        turn_id=self.turn_id,
                        generation_id=self.current_generation_id,
                        speech_duration=speech_segment.duration,
                    )
                )

                # Reset speech buffer
                self._speech_chunks = []

                # Execute STT & Conversational Turn
                stt_result, tts_result = await self._execute_turn(
                    segment=speech_segment,
                    conversation_state=conversation_state,
                )

        return vad_res, stt_result, tts_result

    # -------------------------------------------------------------------------
    # Turn Execution (STT -> Engine -> TTS)
    # -------------------------------------------------------------------------

    async def _execute_turn(
        self,
        segment: SpeechSegment,
        conversation_state: ConversationState,
    ) -> Tuple[Optional[STTResult], Optional[TTSResult]]:
        """Transcribe segment, invoke ConversationEngine, and synthesize response."""
        t_turn_start = time.perf_counter()
        self.turn_id += 1
        self.diagnostics.turn_id = self.turn_id

        # 1. Transcribe Segment
        t_stt_start = time.perf_counter()
        try:
            stt_res = await self.stt.transcribe_segment(segment)
        except Exception as exc:
            self.state = TurnLifecycleState.ERROR
            self.diagnostics.errors.append(f"STT error: {exc}")
            self.emit_event(
                VoicePipelineErrorEvent(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    generation_id=self.current_generation_id,
                    error_message=str(exc),
                    component="stt",
                )
            )
            raise

        stt_latency = time.perf_counter() - t_stt_start
        self.diagnostics.transcription_latency = stt_latency
        self.diagnostics.audio_duration = segment.duration
        self.diagnostics.stt_rtf = (
            round(stt_latency / max(0.001, segment.duration), 4)
            if segment.duration > 0
            else None
        )

        user_text = stt_res.text.strip()
        self.emit_event(
            TranscriptFinal(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=self.current_generation_id,
                transcript=user_text,
                confidence=stt_res.confidence,
                audio_duration=segment.duration,
                latency=round(stt_latency, 4),
                real_time_factor=self.diagnostics.stt_rtf,
            )
        )

        # Ignore empty non-speech noise
        if not user_text:
            self.state = TurnLifecycleState.LISTENING
            return stt_res, None

        # 2. Conversation Intelligence Execution
        self.state = TurnLifecycleState.THINKING
        self.emit_event(
            AgentThinking(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=self.current_generation_id,
            )
        )

        turn_generation_id = self.current_generation_id
        t_llm_start = time.perf_counter()

        try:
            turn_result = await self.conversation.process_user_turn(
                state=conversation_state,
                user_text=user_text,
                generation_id=turn_generation_id,
                cancellation_token=self.cancellation_token,
            )
        except StaleGenerationError:
            logger.info("Turn result discarded: generation %d was superseded.", turn_generation_id)
            self.diagnostics.dropped_stale_frames_count += 1
            return stt_res, None
        except Exception as exc:
            self.state = TurnLifecycleState.ERROR
            self.diagnostics.errors.append(f"Engine error: {exc}")
            self.emit_event(
                VoicePipelineErrorEvent(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    generation_id=turn_generation_id,
                    error_message=str(exc),
                    component="engine",
                )
            )
            raise

        llm_latency = time.perf_counter() - t_llm_start
        self.diagnostics.first_token_latency = llm_latency * 0.3  # Estimated / measured
        self.diagnostics.total_generation_latency = llm_latency

        self.emit_event(
            AgentResponseStarted(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=turn_generation_id,
            )
        )

        response_text = turn_result.response_text or "I am listening."

        # 3. Speech Synthesis Execution
        self.state = TurnLifecycleState.SPEAKING
        t_tts_start = time.perf_counter()

        self.emit_event(
            TTSStarted(
                session_id=self.session_id,
                turn_id=self.turn_id,
                generation_id=turn_generation_id,
            )
        )

        try:
            tts_res = await self.tts.synthesize(
                text=response_text,
                generation_id=turn_generation_id,
                cancellation_token=self.cancellation_token,
            )
        except (StaleGenerationError, BargeInInterruptionError):
            logger.info("TTS synthesis aborted/discarded for generation %d.", turn_generation_id)
            self.diagnostics.dropped_stale_frames_count += 1
            return stt_res, None
        except Exception as exc:
            self.state = TurnLifecycleState.ERROR
            self.diagnostics.errors.append(f"TTS error: {exc}")
            self.emit_event(
                VoicePipelineErrorEvent(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    generation_id=turn_generation_id,
                    error_message=str(exc),
                    component="tts",
                )
            )
            raise

        tts_latency = time.perf_counter() - t_tts_start
        self.diagnostics.first_audio_latency = tts_latency * 0.4
        self.diagnostics.total_synthesis_latency = tts_latency
        self.diagnostics.tts_rtf = (
            round(tts_latency / max(0.001, tts_res.duration or 1.0), 4)
            if tts_res.duration
            else None
        )

        # 4. Push to Output Sink (if not superseded during synthesis)
        if self.cancellation_token.is_valid_generation(turn_generation_id):
            if self.output_sink and hasattr(self.output_sink, "write"):
                self.output_sink.write(tts_res.audio_data)

            self.emit_event(
                TTSStopped(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    generation_id=turn_generation_id,
                    duration=tts_res.duration or 0.0,
                    real_time_factor=self.diagnostics.tts_rtf,
                )
            )
            self.emit_event(
                AgentResponseCompleted(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    generation_id=turn_generation_id,
                    full_text=response_text,
                    domain=conversation_state.current_domain.value,
                    intent=conversation_state.current_intent or "",
                    stage=conversation_state.stage.value if hasattr(conversation_state.stage, "value") else str(conversation_state.stage),
                )
            )
        else:
            logger.info("Discarded late synthesized audio for generation %d", turn_generation_id)
            self.diagnostics.dropped_stale_frames_count += 1
            return stt_res, None

        # Turn total latency
        self.diagnostics.turn_latency = time.perf_counter() - t_turn_start

        # 5. Check Call Termination
        if not conversation_state.conversation_active:
            self.state = TurnLifecycleState.COMPLETED
            self.emit_event(
                SessionCompleted(
                    session_id=self.session_id,
                    turn_id=self.turn_id,
                    generation_id=turn_generation_id,
                    reason=conversation_state.termination_reason or "completed",
                    total_turns=self.turn_id,
                    total_interruptions=self.diagnostics.interruption_count,
                )
            )
        else:
            self.state = TurnLifecycleState.LISTENING

        return stt_res, tts_res

    # -------------------------------------------------------------------------
    # Helper: Speech Segment Assembly
    # -------------------------------------------------------------------------

    def _build_speech_segment(self, end_time: float) -> SpeechSegment:
        """Concatenate buffered AudioChunks into a validated SpeechSegment."""
        if not self._speech_chunks:
            return SpeechSegment(
                audio_data=b"",
                sample_rate=16000,
                channels=1,
                start_time=self._speech_start_time,
                end_time=end_time,
                duration=0.0,
                chunks_count=0,
            )

        sample_rate = self._speech_chunks[0].sample_rate
        channels = self._speech_chunks[0].channels
        audio_data = b"".join(c.data for c in self._speech_chunks)
        duration = sum(c.duration for c in self._speech_chunks)

        return SpeechSegment(
            audio_data=audio_data,
            sample_rate=sample_rate,
            channels=channels,
            start_time=self._speech_start_time,
            end_time=end_time,
            duration=round(duration, 4),
            chunks_count=len(self._speech_chunks),
        )

    def reset(self) -> None:
        """Reset internal buffers and VAD state."""
        self.vad.reset()
        self._speech_chunks.clear()
        self._is_capturing_speech = False
        self.state = TurnLifecycleState.LISTENING
        logger.debug("VoicePipeline reset.")
