"""Voice Session Manager for real-time WebSocket audio streaming and agent orchestration.

Coordinates the complete voice turn lifecycle:
Client (PCM16 Audio) -> AudioProcessor (48kHz->16kHz Resampling) -> Silero VAD ->
Groq STT -> ConversationEngine (Gemini + RAG + MCP) -> Deepgram Streaming TTS ->
Client (Audio Output).

Provides instant barge-in interruption, safe WebSocket transmission, session state recovery,
and graceful disconnect handling.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Callable, Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect

from src.config import get_settings
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.types import CallDirection, DomainType
from src.logging import get_logger
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.telephony.interfaces import CallSessionRepository
from src.telephony.models import CallSummary
from src.voice.audio_services.audio_processor import AudioProcessor
from src.voice.latency_tracker import TurnLatencyTracker
from src.voice.stt.stt_service import STTService
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS

logger = get_logger("voice.session_manager")


async def safe_send(websocket: WebSocket, lock: asyncio.Lock, data: dict) -> bool:
    """Send JSON message over WebSocket safely with concurrency locking."""
    try:
        async with lock:
            await websocket.send_text(json.dumps(data))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as e:
        logger.debug("safe_send error: %s", e)
        return False


async def safe_send_bytes(websocket: WebSocket, lock: asyncio.Lock, data: bytes) -> bool:
    """Send raw binary audio frame over WebSocket safely."""
    try:
        async with lock:
            await websocket.send_bytes(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as e:
        logger.debug("safe_send_bytes error: %s", e)
        return False


class VoiceSessionManager:
    """Manages active WebSocket voice streaming sessions and orchestrates the audio pipeline."""

    def __init__(
        self,
        engine: ConversationEngine,
        state_manager: ConversationStateManager,
        stt_service: Optional[STTService] = None,
        tts_service: Optional[Any] = None,
        default_domain: DomainType = DomainType.INFRA_BUILDER,
        call_repository: Optional[CallSessionRepository] = None,
    ):
        self.engine = engine
        self.state_manager = state_manager
        self.stt_service = stt_service or STTService()
        self.tts_service = tts_service
        self.default_domain = default_domain
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._finalized_sessions: set[str] = set()
        self._finalized_summaries: Dict[str, CallSummary] = {}

        if call_repository is None:
            try:
                from src.telephony.repository import PostgresCallSessionRepository
                self.call_repository = PostgresCallSessionRepository()
            except Exception:
                from src.telephony.repository import MockCallSessionRepository
                self.call_repository = MockCallSessionRepository()
        else:
            self.call_repository = call_repository

    def get_session(self, session_id: str) -> Optional[ConversationState]:
        """Look up active conversation state for a session ID."""
        return self.state_manager.get(session_id)

    def list_active_sessions(self) -> List[str]:
        """Return list of active session IDs."""
        return list(self._active_sessions.keys())

    async def interrupt_speech(self, session_id: str) -> bool:
        """Interrupt active TTS synthesis or agent turn for a session."""
        sess_info = self._active_sessions.get(session_id)
        if not sess_info:
            return False
        sess_info["active_turn_id"] = 0
        task: Optional[asyncio.Task] = sess_info.get("llm_task")
        if task and not task.done():
            task.cancel()
        tts: Optional[DeepgramFluxTTS] = sess_info.get("tts")
        if tts:
            if hasattr(tts, "interrupt"):
                intr_res = tts.interrupt()
                if asyncio.iscoroutine(intr_res):
                    await intr_res
        return True

    async def finalize_session(
        self,
        session_id: str,
        reason: Optional[str] = None,
        state: Optional[ConversationState] = None,
    ) -> Optional[CallSummary]:
        """Finalize voice session, persist interaction dialogue turns and CallSummary idempotently."""
        if session_id in self._finalized_sessions:
            logger.debug("Session %s already finalized. Returning cached summary.", session_id)
            return self._finalized_summaries.get(session_id)

        self._finalized_sessions.add(session_id)

        resolved_state = state or self.get_session(session_id)
        if resolved_state is None and session_id in self._active_sessions:
            resolved_state = self._active_sessions[session_id].get("state")

        if resolved_state is None:
            logger.warning("Cannot finalize unknown voice session %s", session_id)
            return None

        # Resolve session duration & timing
        session_meta = self._active_sessions.get(session_id, {})
        start_time = session_meta.get("started_at") or resolved_state.metadata.start_time or time.time()
        end_time = time.time()
        resolved_state.metadata.end_time = end_time

        disconnect_reason = reason or resolved_state.termination_reason or "caller_disconnected"
        if not resolved_state.termination_reason:
            resolved_state.termination_reason = disconnect_reason
        resolved_state.conversation_active = False

        # Assemble CallSummary from conversation state
        try:
            summary = CallSummary.from_conversation_state(
                state=resolved_state,
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                disconnect_reason=disconnect_reason,
            )
        except Exception as exc:
            logger.warning("Error building CallSummary for session %s: %s", session_id, type(exc).__name__)
            return None

        # Cache summary immediately so callers and status endpoints see it
        self._finalized_summaries[session_id] = summary

        # Non-blocking bounded persistence (turns + summary)
        async def _persist_all() -> None:
            # 1. Persist interaction dialogue turns in chronological order
            for turn in resolved_state.history:
                role_val = turn.role.value if hasattr(turn.role, "value") else str(turn.role)
                speaker = "caller" if role_val == "caller" else ("agent" if role_val == "agent" else role_val)
                text = turn.content or ""
                if not text:
                    continue

                if hasattr(self.call_repository, "save_call_interaction"):
                    await self.call_repository.save_call_interaction(
                        session_id=session_id,
                        speaker=speaker,
                        text=text,
                    )
                elif hasattr(self.call_repository, "postgres_repo") and self.call_repository.postgres_repo is not None:
                    await self.call_repository.postgres_repo.save_call_interaction(
                        session_id=session_id,
                        speaker=speaker,
                        text=text,
                    )

            # 2. Persist CallSummary (lead upsert + summary interaction)
            if self.call_repository is not None:
                await self.call_repository.persist_summary(summary)

        try:
            await asyncio.wait_for(_persist_all(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("Persistence timed out after 3.0s for session %s", session_id)
        except Exception as exc:
            logger.warning("Error persisting session %s to repository: %s", session_id, type(exc).__name__)

        logger.info(
            "Successfully finalized and persisted voice session %s (reason: %s)",
            session_id,
            disconnect_reason,
        )
        return summary



    async def handle_media_stream(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None,
        domain: Optional[DomainType] = None,
        direction: CallDirection = CallDirection.INBOUND,
        caller_phone: Optional[str] = None,
        caller_name: Optional[str] = None,
    ) -> None:
        """Handle full real-time WebSocket voice media stream for a connection."""
        await websocket.accept()

        session_id = session_id or f"voice_session_{int(time.time())}"
        domain = domain or self.default_domain
        connection_alive = True
        ws_lock = asyncio.Lock()

        # Retrieve or initialize persistent ConversationState
        state = self.state_manager.get(session_id)
        if state is None:
            caller_profile = CallerProfile(
                phone=caller_phone,
                name=caller_name,
                email=None,
            )
            init_domain = DomainType.UNKNOWN if direction == CallDirection.INBOUND else domain
            if hasattr(self.state_manager, "create_session_async"):
                state = await self.state_manager.create_session_async(
                    session_id=session_id,
                    domain=init_domain,
                    direction=direction,
                    caller_profile=caller_profile,
                )
            else:
                state = self.state_manager.create_session(
                    session_id=session_id,
                    domain=init_domain,
                    direction=direction,
                    caller_profile=caller_profile,
                )
        else:
            if caller_phone and not state.caller.phone:
                state.caller.phone = caller_phone
            if caller_name and not state.caller.name:
                state.caller.name = caller_name


        # Audio and TTS services for this streaming connection
        settings = get_settings()
        audio_processor = AudioProcessor(
            input_sample_rate=48000,
            output_sample_rate=16000,
            vad_start_threshold=settings.vad_start_threshold,
            vad_end_threshold=settings.vad_end_threshold,
            min_speech_duration_ms=settings.vad_min_speech_duration_ms,
            min_silence_duration_ms=settings.vad_min_silence_duration_ms,
            pre_roll_duration_ms=settings.vad_pre_roll_ms,
            noise_suppression_enabled=settings.noise_suppression_enabled,
            noise_suppression_level=settings.noise_suppression_level,
            high_pass_filter_enabled=settings.high_pass_filter_enabled,
            noise_gate_enabled=settings.noise_gate_enabled,
            noise_gate_threshold_db=settings.noise_gate_threshold_db,
            min_segment_duration_ms=settings.pre_stt_min_duration_ms,
            min_segment_rms=settings.pre_stt_min_rms,
            echo_cancellation_enabled=False,
            auto_gain_control_enabled=False,
        )
        if callable(self.tts_service):
            deepgram_tts = self.tts_service()
        elif isinstance(self.tts_service, DeepgramFluxTTS):
            deepgram_tts = DeepgramFluxTTS(
                model=self.tts_service.model,
                encoding=self.tts_service.encoding,
                sample_rate=self.tts_service.sample_rate,
                api_key=self.tts_service.api_key,
            )
        else:
            deepgram_tts = DeepgramFluxTTS()

        llm_task: Optional[asyncio.Task] = None
        speech_end_at: Optional[float] = None
        turn_counter: int = 0
        current_tracker: Optional[TurnLatencyTracker] = None
        active_turn_id: int = 0
        tts_done_event: asyncio.Event = asyncio.Event()

        # Track active session
        self._active_sessions[session_id] = {
            "started_at": time.time(),
            "websocket": websocket,
            "state": state,
            "tts": deepgram_tts,
            "active_turn_id": active_turn_id,
        }

        # TTS Callbacks sending audio and events to the client
        async def deepgram_audio_callback(audio_bytes: bytes, turn_id: Optional[int] = None):
            if not connection_alive:
                return
            # Reject stale/late audio from an interrupted or superseded turn
            current_active = self._active_sessions.get(session_id, {}).get("active_turn_id", 0)
            if not current_active:
                logger.debug("Discarding stale audio frame for inactive/interrupted turn in session %s", session_id)
                return
            frame_turn = turn_id if turn_id is not None else getattr(deepgram_tts, "current_turn_id", current_active)
            if frame_turn != current_active:
                logger.debug("Discarding stale audio frame from old turn %s (active turn: %s) in session %s", frame_turn, current_active, session_id)
                return
            if current_tracker and current_tracker.t13_first_audio_sent is None:
                current_tracker.mark_first_audio_sent()
            await safe_send_bytes(websocket, ws_lock, audio_bytes)

        async def deepgram_event_callback(event: dict):
            event_type = event.get("type")
            if event_type == "FirstAudio":
                if current_tracker and current_tracker.t12_tts_first_audio is None:
                    current_tracker.mark_tts_first_audio()
                await safe_send(websocket, ws_lock, {"type": "tts_first_audio"})
            elif event_type == "SpeechStarted":
                await safe_send(websocket, ws_lock, {"type": "tts_started"})
            elif event_type in ("SpeechMetadata", "Flushed"):
                tts_done_event.set()
                if current_tracker and current_tracker.t_tts_complete is None:
                    current_tracker.mark_tts_complete()
                await safe_send(websocket, ws_lock, {"type": "tts_complete"})
            elif event_type in ("SpeechInterrupted", "Cleared"):
                tts_done_event.set()
                await safe_send(websocket, ws_lock, {"type": "tts_interrupted"})

        deepgram_tts.audio_callback = deepgram_audio_callback
        deepgram_tts.event_callback = deepgram_event_callback

        try:
            await deepgram_tts.connect()
        except Exception as e:
            logger.warning("[Deepgram] Connect warning: %s", e)

        # Send initial session start event
        await safe_send(websocket, ws_lock, {
            "type": "session_started",
            "session_id": session_id,
            "domain": domain.value,
            "direction": direction.value,
        })

        # If Outbound call and history is empty, agent delivers opening greeting
        if direction == CallDirection.OUTBOUND and len(state.history) == 0:
            turn_counter += 1
            active_turn_id = turn_counter
            if session_id in self._active_sessions:
                self._active_sessions[session_id]["active_turn_id"] = active_turn_id
            tts_done_event.clear()
            llm_task = asyncio.create_task(
                self._process_outbound_greeting(
                    websocket=websocket,
                    ws_lock=ws_lock,
                    state=state,
                    deepgram_tts=deepgram_tts,
                    connection_alive=lambda: connection_alive,
                    turn_id=active_turn_id,
                )
            )

        disconnect_reason: str = "client_disconnected"
        try:
            while connection_alive:
                try:
                    message = await websocket.receive()
                except (WebSocketDisconnect, RuntimeError):
                    logger.info("WebSocket disconnected for session %s", session_id)
                    disconnect_reason = "client_disconnected"
                    break
                except Exception as exc:
                    logger.warning("WebSocket receive error in %s: %s", session_id, exc)
                    disconnect_reason = f"receive_error_{type(exc).__name__}"
                    break

                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
                    disconnect_reason = "client_disconnected"
                    break

                # Handle text control messages or Twilio Media Stream JSON messages
                audio_bytes = message.get("bytes")
                text_data = message.get("text")
                if text_data:
                    try:
                        control_msg = json.loads(text_data)
                        event_type = control_msg.get("event") or control_msg.get("type")
                        if event_type in ("hangup", "session_end", "stop"):
                            logger.info("Client requested hangup for session %s", session_id)
                            disconnect_reason = f"client_{event_type}"
                            break
                        elif event_type == "ping":
                            await safe_send(websocket, ws_lock, {"type": "pong"})
                            continue
                        elif event_type == "start":
                            stream_info = control_msg.get("start", {})
                            stream_sid = stream_info.get("streamSid") or control_msg.get("streamSid")
                            if stream_sid:
                                logger.info("MediaStream started: %s", stream_sid)
                            continue
                        elif event_type == "media":
                            media_data = control_msg.get("media", {})
                            payload_b64 = media_data.get("payload")
                            if payload_b64:
                                audio_bytes = base64.b64decode(payload_b64)
                    except Exception as exc:
                        logger.debug("Failed parsing text message: %s", exc)

                if not audio_bytes:
                    continue

                try:
                    vad_result = audio_processor.process_with_vad(audio_bytes)
                except Exception as vad_err:
                    logger.debug("VAD processing error: %s", vad_err)
                    continue

                # 1. Instant Barge-In
                if vad_result.get("speech_started"):
                    turn_counter += 1
                    active_turn_id = 0
                    if session_id in self._active_sessions:
                        self._active_sessions[session_id]["active_turn_id"] = 0
                    tts_done_event.set()

                    current_tracker = TurnLatencyTracker(
                        turn_id=turn_counter,
                        session_id=session_id,
                        domain=state.current_domain.value,
                    )
                    current_tracker.mark_speech_start()

                    if llm_task is not None and not llm_task.done():
                        llm_task.cancel()
                        llm_task = None
                        logger.info("Barge-in: cancelled active agent turn for session %s", session_id)

                    if hasattr(deepgram_tts, "interrupt"):
                        intr_res = deepgram_tts.interrupt()
                        if asyncio.iscoroutine(intr_res):
                            await intr_res
                    await safe_send(websocket, ws_lock, {"type": "interrupt"})
                    await safe_send(websocket, ws_lock, {"type": "speech_started"})

                # 2. Check if speech ended
                if not vad_result.get("speech_ended"):
                    continue

                speech_end_at = time.perf_counter()
                if current_tracker is None:
                    turn_counter += 1
                    current_tracker = TurnLatencyTracker(
                        turn_id=turn_counter,
                        session_id=session_id,
                        domain=state.current_domain.value,
                    )
                current_tracker.mark_speech_end(speech_end_at)
                current_tracker.mark_stt_start()

                speech_audio = vad_result.get("speech_audio")
                if not speech_audio:
                    continue

                # 3. Speech-to-Text (Non-blocking async transcription)
                try:
                    transcript = (await self.stt_service.transcribe_pcm16_async(speech_audio)).strip()
                except Exception as e:
                    logger.warning("[STT] Transcription error: %s", e)
                    continue

                current_tracker.mark_stt_end()

                if not transcript:
                    continue

                await safe_send(websocket, ws_lock, {"type": "transcript", "text": transcript})

                # 4. Launch Agent Turn
                if llm_task is not None and not llm_task.done():
                    llm_task.cancel()
                    llm_task = None

                turn_counter += 1
                active_turn_id = turn_counter
                if session_id in self._active_sessions:
                    self._active_sessions[session_id]["active_turn_id"] = active_turn_id
                tts_done_event.clear()

                llm_task = asyncio.create_task(
                    self._process_agent_turn(
                        websocket=websocket,
                        ws_lock=ws_lock,
                        state=state,
                        transcript=transcript,
                        deepgram_tts=deepgram_tts,
                        connection_alive=lambda: connection_alive,
                        tracker=current_tracker,
                        turn_id=active_turn_id,
                        tts_done_event=tts_done_event,
                    )
                )
                if session_id in self._active_sessions:
                    self._active_sessions[session_id]["llm_task"] = llm_task

        finally:
            connection_alive = False
            if llm_task is not None and not llm_task.done():
                llm_task.cancel()

            eff_reason = disconnect_reason
            if state and not state.conversation_active and state.termination_reason:
                eff_reason = state.termination_reason

            try:
                await self.finalize_session(session_id=session_id, reason=eff_reason, state=state)
            except Exception as e:
                logger.warning("Error during session finalization for %s: %s", session_id, type(e).__name__)

            try:
                await deepgram_tts.close()
            except Exception:
                pass

            try:
                audio_processor.close()
            except Exception:
                pass

            self._active_sessions.pop(session_id, None)
            logger.info("Cleaned up voice session %s", session_id)

    async def _process_agent_turn(
        self,
        websocket: WebSocket,
        ws_lock: asyncio.Lock,
        state: ConversationState,
        transcript: str,
        deepgram_tts: DeepgramFluxTTS,
        connection_alive: Callable[[], bool],
        tracker: Optional[TurnLatencyTracker] = None,
        turn_id: int = 1,
        tts_done_event: Optional[asyncio.Event] = None,
        enable_grounded_streaming: bool = True,
    ) -> None:
        """Process an agent turn and stream response speech to TTS and WebSocket."""
        session_id = state.session_id
        if session_id in self._active_sessions:
            self._active_sessions[session_id]["active_turn_id"] = turn_id

        if hasattr(deepgram_tts, "reset_turn"):
            try:
                deepgram_tts.reset_turn(turn_id=turn_id)
            except TypeError:
                deepgram_tts.reset_turn()
        await safe_send(websocket, ws_lock, {"type": "agent_started"})

        streamed_phrases: List[str] = []

        async def on_grounded_chunk(phrase: str) -> None:
            if not connection_alive():
                return
            # Fast check: ensure this turn has not been invalidated by barge-in
            current_active = self._active_sessions.get(session_id, {}).get("active_turn_id", turn_id)
            if current_active != turn_id:
                raise asyncio.CancelledError(f"Turn {turn_id} superseded by barge-in.")

            phrase_clean = phrase.strip()
            if not phrase_clean:
                return

            streamed_phrases.append(phrase_clean)
            if tracker and tracker.t11_tts_start is None:
                tracker.mark_tts_start()

            phrase_to_send = phrase_clean + " "
            if hasattr(deepgram_tts, "send_text"):
                try:
                    send_coro = deepgram_tts.send_text(phrase_to_send, turn_id=turn_id)
                except TypeError:
                    send_coro = deepgram_tts.send_text(phrase_to_send)
                if asyncio.iscoroutine(send_coro):
                    await send_coro
            await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": phrase_to_send})

        try:
            # Execute turn through core ConversationEngine with live streaming callback
            t0 = time.perf_counter()
            grounded_cb = on_grounded_chunk if enable_grounded_streaming else None
            turn_result: EngineTurnResult = await self.engine.process_user_turn(
                state=state,
                user_message=transcript,
                tracker=tracker,
                on_grounded_chunk=grounded_cb,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            response_text = (turn_result.response_text or "").strip()
            if not response_text:
                response_text = "I apologize, but I could not find information regarding that inquiry."

            if not connection_alive():
                return

            # Check after generation completes if turn was interrupted
            current_active = self._active_sessions.get(session_id, {}).get("active_turn_id", turn_id)
            if current_active != turn_id:
                raise asyncio.CancelledError(f"Turn {turn_id} superseded by barge-in.")

            if tracker:
                tracker.domain = state.current_domain.value
                tracker.intent = turn_result.decision.detected_intent or ""

            # Send telemetry to client
            telemetry_data = {
                "type": "agent_telemetry",
                "route": turn_result.decision.detected_intent or "general",
                "confidence": 1.0,
                "stage": state.stage.value if hasattr(state.stage, "value") else str(state.stage),
                "domain": state.current_domain.value,
                "tool_name": turn_result.tool_result.tool_name if turn_result.tool_result else None,
                "tool_result": turn_result.tool_result.data if turn_result.tool_result else None,
                "citations": turn_result.grounded_citations,
                "latency_ms": round(elapsed_ms, 2),
            }
            if tracker:
                telemetry_data["latency_metrics"] = tracker.to_dict()

            await safe_send(websocket, ws_lock, telemetry_data)

            if streamed_phrases:
                # Streaming mode: phrases were already delivered incrementally during LLM synthesis.
                # Flush the TTS stream once to finalize audio synthesis.
                if connection_alive() and hasattr(deepgram_tts, "flush"):
                    flush_coro = deepgram_tts.flush()
                    if asyncio.iscoroutine(flush_coro):
                        await flush_coro
            else:
                # Non-streaming mode (e.g., greetings, non-RAG turns, tool responses, unsupported streaming):
                # Deliver words via standard burst chunking to Deepgram TTS.
                words = response_text.split()
                if words:
                    if tracker:
                        tracker.mark_tts_start()
                    first_burst = " ".join(words[:3]) + " "
                    if hasattr(deepgram_tts, "send_text"):
                        try:
                            send_coro = deepgram_tts.send_text(first_burst, turn_id=turn_id)
                        except TypeError:
                            send_coro = deepgram_tts.send_text(first_burst)
                        if asyncio.iscoroutine(send_coro):
                            await send_coro
                    await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": first_burst})

                    for i in range(3, len(words), 4):
                        if not connection_alive():
                            return
                        current_active = self._active_sessions.get(session_id, {}).get("active_turn_id", turn_id)
                        if current_active != turn_id:
                            raise asyncio.CancelledError(f"Turn {turn_id} superseded by barge-in.")
                        chunk = " ".join(words[i : i + 4]) + " "
                        if hasattr(deepgram_tts, "send_text"):
                            try:
                                send_coro = deepgram_tts.send_text(chunk, turn_id=turn_id)
                            except TypeError:
                                send_coro = deepgram_tts.send_text(chunk)
                            if asyncio.iscoroutine(send_coro):
                                await send_coro
                        await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": chunk})

                if connection_alive() and hasattr(deepgram_tts, "flush"):
                    flush_coro = deepgram_tts.flush()
                    if asyncio.iscoroutine(flush_coro):
                        await flush_coro

            # Wait boundedly for TTS audio output completion if event is provided
            if tts_done_event and connection_alive():
                try:
                    await asyncio.wait_for(tts_done_event.wait(), timeout=5.0)
                    if tracker and tracker.t_tts_complete is None:
                        tracker.mark_tts_complete()
                except asyncio.TimeoutError:
                    logger.debug("TTS audio completion wait timed out for turn %s", turn_id)
                    if tracker and tracker.t_tts_complete is None:
                        tracker.mark_tts_complete()
            elif tracker and tracker.t_tts_complete is None:
                tracker.mark_tts_complete()

            await safe_send(websocket, ws_lock, {"type": "llm_complete", "text": response_text})

            if tracker:
                tracker.log_summary()

            # Check if conversation was closed by caller or agent
            if not state.conversation_active:
                await safe_send(websocket, ws_lock, {
                    "type": "session_ended",
                    "reason": state.termination_reason or "completed",
                })

        except asyncio.CancelledError:
            logger.info("Agent turn task cancelled due to barge-in (turn %s).", turn_id)
            raise
        except Exception as e:
            logger.exception("Error in agent turn: %s", e)
            await safe_send(websocket, ws_lock, {"type": "error", "message": "Agent processing failed."})

    async def _process_outbound_greeting(
        self,
        websocket: WebSocket,
        ws_lock: asyncio.Lock,
        state: ConversationState,
        deepgram_tts: DeepgramFluxTTS,
        connection_alive: Callable[[], bool],
        turn_id: int = 1,
    ) -> None:
        """Execute and stream initial outbound campaign greeting."""
        if hasattr(deepgram_tts, "reset_turn"):
            try:
                deepgram_tts.reset_turn(turn_id=turn_id)
            except TypeError:
                deepgram_tts.reset_turn()
        await safe_send(websocket, ws_lock, {"type": "agent_started"})
        try:
            t0 = time.perf_counter()
            outbound_result = await self.engine.start_outbound_conversation(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            response_text = (outbound_result.response_text or "").strip()
            if not response_text:
                contact = state.caller.name or state.contact_name
                greeting_prefix = f"Hello {contact}, " if contact else "Hello, "
                response_text = f"{greeting_prefix}this is Infra Builder Real Estate following up on your property inquiry. How are you doing today?"

            if not connection_alive():
                return

            await safe_send(websocket, ws_lock, {
                "type": "agent_telemetry",
                "route": "outbound_greeting",
                "confidence": 1.0,
                "stage": state.stage.value if hasattr(state.stage, "value") else str(state.stage),
                "domain": state.current_domain.value,
                "latency_ms": round(elapsed_ms, 2),
            })

            words = response_text.split()
            if words:
                first_burst = " ".join(words[:3]) + " "
                if hasattr(deepgram_tts, "send_text"):
                    try:
                        send_coro = deepgram_tts.send_text(first_burst, turn_id=turn_id)
                    except TypeError:
                        send_coro = deepgram_tts.send_text(first_burst)
                    if asyncio.iscoroutine(send_coro):
                        await send_coro
                await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": first_burst})

                for i in range(3, len(words), 4):
                    if not connection_alive():
                        return
                    chunk = " ".join(words[i : i + 4]) + " "
                    if hasattr(deepgram_tts, "send_text"):
                        try:
                            send_coro = deepgram_tts.send_text(chunk, turn_id=turn_id)
                        except TypeError:
                            send_coro = deepgram_tts.send_text(chunk)
                        if asyncio.iscoroutine(send_coro):
                            await send_coro
                    await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": chunk})

            if connection_alive() and hasattr(deepgram_tts, "flush"):
                flush_coro = deepgram_tts.flush()
                if asyncio.iscoroutine(flush_coro):
                    await flush_coro

            await safe_send(websocket, ws_lock, {"type": "llm_complete", "text": response_text})
        except asyncio.CancelledError:
            logger.info("Outbound greeting task cancelled due to barge-in.")
            raise
        except Exception as e:
            logger.exception("Error in outbound greeting: %s", e)
            await safe_send(websocket, ws_lock, {"type": "error", "message": "Failed to start outbound greeting."})
