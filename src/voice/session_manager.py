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
from src.voice.audio_services.audio_processor import AudioProcessor
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
        default_domain: DomainType = DomainType.EDUSAAS,
    ):
        self.engine = engine
        self.state_manager = state_manager
        self.stt_service = stt_service or STTService()
        self.tts_service = tts_service
        self.default_domain = default_domain
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

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
        tts: Optional[DeepgramFluxTTS] = sess_info.get("tts")
        if tts:
            await tts.interrupt()
        return True

    async def handle_media_stream(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None,
        domain: Optional[DomainType] = None,
        direction: CallDirection = CallDirection.INBOUND,
        caller_phone: Optional[str] = None,
        caller_name: Optional[str] = None,
        already_accepted: bool = False,
    ) -> None:
        """Handle full real-time WebSocket voice media stream for a connection."""
        if not already_accepted and getattr(websocket.client_state, "name", "") != "CONNECTED":
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
            state = self.state_manager.create_session(
                session_id=session_id,
                domain=init_domain,
                direction=direction,
                caller_profile=caller_profile,
            )

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

        # Track active session
        self._active_sessions[session_id] = {
            "started_at": time.time(),
            "websocket": websocket,
            "state": state,
            "tts": deepgram_tts,
        }

        # TTS Callbacks sending audio and events to the client
        async def deepgram_audio_callback(audio_bytes: bytes):
            if not connection_alive:
                return
            await safe_send_bytes(websocket, ws_lock, audio_bytes)

        async def deepgram_event_callback(event: dict):
            event_type = event.get("type")
            if event_type == "FirstAudio":
                await safe_send(websocket, ws_lock, {"type": "tts_first_audio"})
            elif event_type == "SpeechStarted":
                await safe_send(websocket, ws_lock, {"type": "tts_started"})
            elif event_type == "SpeechMetadata":
                await safe_send(websocket, ws_lock, {"type": "tts_complete"})
            elif event_type == "SpeechInterrupted":
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
            llm_task = asyncio.create_task(
                self._process_outbound_greeting(
                    websocket=websocket,
                    ws_lock=ws_lock,
                    state=state,
                    deepgram_tts=deepgram_tts,
                    connection_alive=lambda: connection_alive,
                )
            )

        try:
            while connection_alive:
                try:
                    message = await websocket.receive()
                except (WebSocketDisconnect, RuntimeError):
                    logger.info("WebSocket disconnected for session %s", session_id)
                    break
                except Exception as exc:
                    logger.warning("WebSocket receive error in %s: %s", session_id, exc)
                    break

                msg_type = message.get("type")
                if msg_type == "websocket.disconnect":
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
                    if llm_task is not None and not llm_task.done():
                        llm_task.cancel()
                        llm_task = None
                        logger.info("Barge-in: cancelled active agent turn for session %s", session_id)

                    await deepgram_tts.interrupt()
                    await safe_send(websocket, ws_lock, {"type": "interrupt"})
                    await safe_send(websocket, ws_lock, {"type": "speech_started"})

                # 2. Check if speech ended
                if not vad_result.get("speech_ended"):
                    continue

                speech_end_at = time.perf_counter()
                speech_audio = vad_result.get("speech_audio")
                if not speech_audio:
                    continue

                # 3. Speech-to-Text
                try:
                    transcript = self.stt_service.transcribe_pcm16(speech_audio).strip()
                except Exception as e:
                    logger.warning("[STT] Transcription error: %s", e)
                    continue

                if not transcript:
                    continue

                await safe_send(websocket, ws_lock, {"type": "transcript", "text": transcript})

                # 4. Launch Agent Turn
                llm_task = asyncio.create_task(
                    self._process_agent_turn(
                        websocket=websocket,
                        ws_lock=ws_lock,
                        state=state,
                        transcript=transcript,
                        deepgram_tts=deepgram_tts,
                        connection_alive=lambda: connection_alive,
                    )
                )

        finally:
            connection_alive = False
            if llm_task is not None and not llm_task.done():
                llm_task.cancel()

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
    ) -> None:
        """Process an agent turn and stream response speech to TTS and WebSocket."""
        deepgram_tts.reset_turn()
        await safe_send(websocket, ws_lock, {"type": "agent_started"})

        try:
            # Execute turn through Vayvora's core ConversationEngine
            t0 = time.perf_counter()
            turn_result: EngineTurnResult = await self.engine.process_user_turn(
                state=state,
                user_message=transcript,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            response_text = (turn_result.response_text or "").strip()
            if not response_text:
                response_text = "I apologize, but I could not find information regarding that inquiry."

            if not connection_alive():
                return

            # Send telemetry to client
            await safe_send(websocket, ws_lock, {
                "type": "agent_telemetry",
                "route": turn_result.decision.detected_intent or "general",
                "confidence": 1.0,
                "stage": state.stage.value if hasattr(state.stage, "value") else str(state.stage),
                "domain": state.current_domain.value,
                "tool_name": turn_result.tool_result.tool_name if turn_result.tool_result else None,
                "tool_result": turn_result.tool_result.data if turn_result.tool_result else None,
                "citations": turn_result.grounded_citations,
                "latency_ms": round(elapsed_ms, 2),
            })

            # Stream words to Deepgram TTS with 3-word instant burst for minimal latency
            words = response_text.split()
            if words:
                first_burst = " ".join(words[:3]) + " "
                await deepgram_tts.send_text(first_burst)
                await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": first_burst})

                for i in range(3, len(words), 4):
                    if not connection_alive():
                        return
                    chunk = " ".join(words[i : i + 4]) + " "
                    await deepgram_tts.send_text(chunk)
                    await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": chunk})

            if connection_alive():
                await deepgram_tts.flush()

            await safe_send(websocket, ws_lock, {"type": "llm_complete", "text": response_text})

            # Check if conversation was closed by caller or agent
            if not state.conversation_active:
                await safe_send(websocket, ws_lock, {
                    "type": "session_ended",
                    "reason": state.termination_reason or "completed",
                })

        except asyncio.CancelledError:
            logger.info("Agent turn task cancelled due to barge-in.")
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
    ) -> None:
        """Execute and stream initial outbound campaign greeting."""
        deepgram_tts.reset_turn()
        await safe_send(websocket, ws_lock, {"type": "agent_started"})
        try:
            t0 = time.perf_counter()
            outbound_result = await self.engine.start_outbound_conversation(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            response_text = (outbound_result.response_text or "").strip()
            if not response_text:
                if state.current_domain == DomainType.EDUSAAS:
                    response_text = "Hi, this is the EduSaaS admissions team. How can I help you today?"
                else:
                    response_text = "Hello, this is the Vayvora team following up on your inquiry. How are you doing today?"

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
                await deepgram_tts.send_text(first_burst)
                await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": first_burst})

                for i in range(3, len(words), 4):
                    if not connection_alive():
                        return
                    chunk = " ".join(words[i : i + 4]) + " "
                    await deepgram_tts.send_text(chunk)
                    await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": chunk})

            if connection_alive():
                await deepgram_tts.flush()

            await safe_send(websocket, ws_lock, {"type": "llm_complete", "text": response_text})
        except asyncio.CancelledError:
            logger.info("Outbound greeting task cancelled due to barge-in.")
        except Exception as e:
            logger.exception("Error in outbound greeting: %s", e)
            await safe_send(websocket, ws_lock, {"type": "error", "message": "Failed to start outbound greeting."})
