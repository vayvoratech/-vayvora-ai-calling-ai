import asyncio
import json
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from src.voice.audio_services.audio_processor import AudioProcessor
from src.voice.stt.stt_service import STTService
from src.llm.gemini_service import GeminiService
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS


router = APIRouter()

PUBLIC_WS_URL = (
    "wss://yorkshire-webpage-belfast-graphical"
    ".trycloudflare.com/media-stream"
)

# CPU execution on Alder Lake avoids Vulkan UMA overhead on Intel Iris Xe
stt_service = STTService(
    backend="cpu",
    n_threads=8
)

gemini_service = GeminiService(
    model="gemini-3.5-flash-lite"
)


async def safe_send(
    websocket: WebSocket,
    lock: asyncio.Lock,
    data: dict
) -> bool:
    try:
        async with lock:
            await websocket.send_text(json.dumps(data))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as e:
        print("WebSocket send error:", type(e).__name__, str(e))
        return False


async def safe_send_bytes(
    websocket: WebSocket,
    lock: asyncio.Lock,
    data: bytes
) -> bool:
    try:
        async with lock:
            await websocket.send_bytes(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as e:
        print("Binary send error:", type(e).__name__, str(e))
        return False


@router.post("/voice")
async def voice():
    twiml = f"""
<Response>
    <Connect>
        <Stream url="{PUBLIC_WS_URL}" />
    </Connect>
</Response>
"""
    return Response(
        content=twiml,
        media_type="application/xml"
    )


@router.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()

    connection_alive = True
    ws_lock = asyncio.Lock()

    audio_processor = AudioProcessor(
        input_sample_rate=48000,
        output_sample_rate=16000,
        vad_threshold=0.5
    )
    deepgram_tts = DeepgramFluxTTS()

    conversation_history: list[dict] = []
    speech_end_at = None
    llm_task = None

    async def deepgram_audio_callback(audio_bytes: bytes):
        if not connection_alive:
            return
        await safe_send_bytes(websocket, ws_lock, audio_bytes)

    async def deepgram_event_callback(event: dict):
        event_type = event.get("type")

        if event_type == "FirstAudio":
            now = time.perf_counter()
            print()
            print("********************************")
            print("DEEPGRAM FIRST AUDIO -> APPLICATION")
            print("********************************")
            if speech_end_at is not None:
                print(f"Speech End -> First Audio: {now - speech_end_at:.3f}s")
            print("********************************\n")
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
        print("Unable to connect Deepgram:", type(e).__name__, str(e))
        connection_alive = False

    try:
        while connection_alive:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                connection_alive = False
                break
            except Exception:
                connection_alive = False
                break

            if message.get("type") == "websocket.disconnect":
                connection_alive = False
                break

            audio_bytes = message.get("bytes")
            if not audio_bytes:
                continue

            try:
                vad_result = audio_processor.process_with_vad(audio_bytes)
            except Exception:
                continue

            # Instant barge-in cancellation
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
            except Exception:
                continue

            if not transcript:
                continue

            print(">>> USER:", transcript)

            ok = await safe_send(
                websocket,
                ws_lock,
                {"type": "transcript", "text": transcript}
            )
            if not ok:
                connection_alive = False
                break

            llm_task = asyncio.create_task(
                process_llm_turn(
                    websocket=websocket,
                    ws_lock=ws_lock,
                    history=conversation_history,
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


# ============================================================
# BALANCED-PACING STREAMING (STEADY NATURAL CADENCE)
# ============================================================

async def process_llm_turn(
    websocket: WebSocket,
    ws_lock: asyncio.Lock,
    history: list[dict],
    transcript: str,
    deepgram_tts: DeepgramFluxTTS,
    connection_state,
    speech_end_at=None
):
    turn_start = time.perf_counter()
    full_response = ""
    first_token_received = False

    history.append({"role": "user", "text": transcript})
    if len(history) > 4:
        del history[:len(history) - 4]

    turn_snapshot = list(history)

    try:
        deepgram_tts.reset_turn()
        await safe_send(websocket, ws_lock, {"type": "llm_started"})

        stream_buffer = ""

        async for chunk in gemini_service.generate_stream_turn(turn_snapshot):
            if not connection_state():
                return

            if not chunk:
                continue

            if not first_token_received:
                first_token_received = True
                now = time.perf_counter()
                print(f"\nGemini FIRST TOKEN: {now - turn_start:.3f}s")
                if speech_end_at is not None:
                    print(f"Speech End -> Gemini First Token: {now - speech_end_at:.3f}s")

            full_response += chunk
            stream_buffer += chunk

            # Real-time token display on frontend
            await safe_send(websocket, ws_lock, {"type": "llm_chunk", "text": chunk})

            # Check for natural punctuation or a 4-word phrase threshold
            has_punct = any(p in stream_buffer for p in [".", "?", "!", ",", ";", ":"])
            words = stream_buffer.strip().split()

            if has_punct or (len(words) >= 4 and " " in stream_buffer):
                if has_punct:
                    # Cut cleanly at the punctuation mark
                    p_idx = max(stream_buffer.rfind(p) for p in [".", "?", "!", ",", ";", ":"] if p in stream_buffer)
                    phrase = stream_buffer[:p_idx + 1].strip()
                    stream_buffer = stream_buffer[p_idx + 1:].lstrip()
                else:
                    # Cut cleanly at the last full word boundary
                    space_idx = stream_buffer.rfind(" ")
                    phrase = stream_buffer[:space_idx].strip()
                    stream_buffer = stream_buffer[space_idx:].lstrip()

                if phrase:
                    # Sent naturally without artificial comma injection
                    await deepgram_tts.send_text(phrase + " ")

        # Dispatch any remaining trailing words
        if stream_buffer.strip() and connection_state():
            await deepgram_tts.send_text(stream_buffer.strip() + " ")

        # Signal end of turn to Deepgram
        if connection_state():
            await deepgram_tts.flush()

        history.append({"role": "model", "text": full_response})
        print(f"Turn Complete: {time.perf_counter() - turn_start:.3f}s | Output: {full_response}")

        await safe_send(
            websocket,
            ws_lock,
            {"type": "llm_complete", "text": full_response}
        )

    except asyncio.CancelledError:
        if history and history[-1]["role"] == "user":
            history.pop()
        raise
    except Exception as e:
        if history and history[-1]["role"] == "user":
            history.pop()
        print("LLM ERROR:", type(e).__name__, str(e))
        await safe_send(
            websocket,
            ws_lock,
            {"type": "error", "message": "Language model failed."}
        )