import asyncio
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from api.dependencies import get_agent_runtime
from src.voice.audio_services.audio_processor import AudioProcessor
from src.voice.stt.stt_service import STTService
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS

router = APIRouter(tags=["Voice Streaming"])

PUBLIC_WS_URL = "wss://your-domain.com/media-stream"

# Initialize STT on CPU
stt_service = STTService(
    model="whisper-large-v3-turbo",
    language="en"
)


async def safe_send(websocket: WebSocket, lock: asyncio.Lock, data: dict) -> bool:
    try:
        async with lock:
            await websocket.send_text(json.dumps(data))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception:
        return False


async def safe_send_bytes(websocket: WebSocket, lock: asyncio.Lock, data: bytes) -> bool:
    try:
        async with lock:
            await websocket.send_bytes(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception:
        return False


@router.post("/api/v1/voice")
async def telephony_webhook():
    """TwiML / Telephony connector endpoint"""
    twiml = f"""
<Response>
    <Connect>
        <Stream url="{PUBLIC_WS_URL}" />
    </Connect>
</Response>
"""
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/media-stream")
async def voice_media_stream(websocket: WebSocket):
    await websocket.accept()

    connection_alive = True
    ws_lock = asyncio.Lock()

    audio_processor = AudioProcessor(
        input_sample_rate=48000,
        output_sample_rate=16000,
        vad_start_threshold=0.35,
        vad_end_threshold=0.20,
        min_silence_duration_ms=250
    )
    deepgram_tts = DeepgramFluxTTS()

    conversation_history: list[dict] = []
    conversation_slots: dict = {}
    session_id = f"voice_session_{int(time.time())}"

    speech_end_at = None
    llm_task = None

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
        print("[Deepgram] Connect error:", e)
        connection_alive = False

    try:
        while connection_alive:
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, Exception):
                break

            if message.get("type") == "websocket.disconnect":
                break

            audio_bytes = message.get("bytes")
            if not audio_bytes:
                continue

            try:
                vad_result = audio_processor.process_with_vad(audio_bytes)
            except Exception:
                continue

            # Instant Barge-In
            if vad_result.get("speech_started"):
                if llm_task is not None and not llm_task.done():
                    llm_task.cancel()
                    llm_task = None

                await deepgram_tts.interrupt()
                await safe_send(websocket, ws_lock, {"type": "interrupt"})
                await safe_send(websocket, ws_lock, {"type": "speech_started"})

            if not vad_result.get("speech_ended"):
                continue

            speech_end_at = time.perf_counter()
            speech_audio = vad_result.get("speech_audio")
            if not speech_audio:
                continue

            try:
                transcript = stt_service.transcribe_pcm16(speech_audio).strip()
            except Exception as e:
                print("[STT] Error:", e)
                continue

            if not transcript:
                continue

            await safe_send(websocket, ws_lock, {"type": "transcript", "text": transcript})

            llm_task = asyncio.create_task(
                process_agent_turn(
                    websocket=websocket,
                    ws_lock=ws_lock,
                    history=conversation_history,
                    slots=conversation_slots,
                    session_id=session_id,
                    transcript=transcript,
                    deepgram_tts=deepgram_tts,
                    connection_state=lambda: connection_alive,
                    speech_end_at=speech_end_at
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
            audio_processor.reset()
        except Exception:
            pass


async def process_agent_turn(
    websocket: WebSocket,
    ws_lock: asyncio.Lock,
    history: list[dict],
    slots: dict,
    session_id: str,
    transcript: str,
    deepgram_tts: DeepgramFluxTTS,
    connection_state,
    speech_end_at=None
):
    turn_start = time.perf_counter()
    deepgram_tts.reset_turn()
    await safe_send(websocket, ws_lock, {"type": "agent_started"})

    agent = get_agent_runtime()

    state = {
        "session_id": session_id,
        "user_input": transcript,
        "tenant_id": "default",
        "messages": list(history),
        "slots": slots,
    }

    try:
        result = await agent.run(state)
        response_text = result.get("response", "").strip()

        if "slots" in result and isinstance(result["slots"], dict):
            slots.update(result["slots"])

        if not response_text:
            response_text = "I'm sorry, I couldn't find an answer to that."

        # Send telemetry to client/gateway
        await safe_send(websocket, ws_lock, {
            "type": "agent_telemetry",
            "route": result.get("route", "unknown"),
            "confidence": result.get("route_confidence", 0.0),
            "tool_name": result.get("tool_name"),
            "tool_result": result.get("tool_result")
        })

        if not connection_state():
            return

        words = response_text.split()
        if words:
            # 3-word instant burst reduces time-to-first-audio
            first_burst = " ".join(words[:3]) + " "
            await deepgram_tts.send_text(first_burst)
            await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": first_burst})

            for i in range(3, len(words), 4):
                if not connection_state():
                    return
                chunk = " ".join(words[i:i + 4]) + " "
                await deepgram_tts.send_text(chunk)
                await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": chunk})

        if connection_state():
            await deepgram_tts.flush()

        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": response_text})

        if len(history) > 8:
            del history[:len(history) - 8]

        await safe_send(websocket, ws_lock, {"type": "llm_complete", "text": response_text})

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print("[Agent Turn] Error:", e)
        await safe_send(websocket, ws_lock, {"type": "error", "message": "Agent turn failed."})