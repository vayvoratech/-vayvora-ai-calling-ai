"""Deepgram streaming Text-to-Speech (TTS) service using WebSocket protocol.



Follows the reference implementation from the Kiran project:

- Direct WebSocket streaming to wss://api.deepgram.com/{version}/speak

- Models: aura-asteria-en (v1/speak) or Deepgram Aura / Flux

- Audio encoding: linear16, 48kHz Mono

- 3-word instant burst + streaming word delivery

- Barge-in cancellation via Clear/Interrupt control frames

- FirstAudio latency tracking and event callback dispatch

"""



from __future__ import annotations



import asyncio

import json

import os

import time

from typing import Any, AsyncIterator, Callable, Optional, Union

import websockets



from src.core.interfaces import TTSProvider

from src.logging import get_logger



logger = get_logger("voice.tts")





class DeepgramFluxTTS(TTSProvider):

    """Real-time streaming TTS client for Deepgram Aura/Flux TTS."""



    def __init__(

        self,

        model: Optional[str] = None,

        encoding: Optional[str] = None,

        sample_rate: Optional[int] = None,

        api_key: Optional[str] = None,

    ):

        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")



        # Default to ultra-fast Aura voice model

        self.model = (

            model

            or os.getenv("DEEPGRAM_TTS_MODEL")

            or "aura-asteria-en"

        )



        self.encoding = (

            encoding

            or os.getenv("DEEPGRAM_TTS_ENCODING")

            or "linear16"

        )



        self.sample_rate = int(

            sample_rate

            or os.getenv("DEEPGRAM_TTS_SAMPLE_RATE")

            or "48000"

        )



        self.websocket = None

        self.receive_task: Optional[asyncio.Task] = None

        self.connected = False

        self._mock_mode = not bool(self.api_key)



        self.audio_callback: Optional[Callable[[bytes], Any]] = None

        self.event_callback: Optional[Callable[[dict], Any]] = None



        self.turn_started_at: Optional[float] = None

        self.first_speak_sent_at: Optional[float] = None

        self.first_audio_received = False

        self.first_audio_event_reported = False

        self._accumulated_text: list[str] = []

        # Automatic reconnect state.
        self._closing = False
        self._connect_lock = asyncio.Lock()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = max(0, int(os.getenv("DEEPGRAM_TTS_MAX_RECONNECT_ATTEMPTS", "0")))
        self._reconnect_base_delay = max(0.1, float(os.getenv("DEEPGRAM_TTS_RECONNECT_BASE_DELAY", "1.0")))
        self._reconnect_max_delay = max(
            self._reconnect_base_delay,
            float(os.getenv("DEEPGRAM_TTS_RECONNECT_MAX_DELAY", "8.0")),
        )



    def is_open(self) -> bool:

        """Version-agnostic check if WebSocket connection is open and active."""

        if self._mock_mode:

            return self.connected

        if not self.websocket or not self.connected:

            return False

        # 1. Closed attribute

        if hasattr(self.websocket, "closed") and self.websocket.closed:

            return False

        # 2. Open attribute

        if hasattr(self.websocket, "open") and not self.websocket.open:

            return False

        # 3. Close code attribute

        if hasattr(self.websocket, "close_code") and self.websocket.close_code is not None:

            return False

        # 4. State enum attribute (websockets library)

        if hasattr(self.websocket, "state"):

            st = getattr(self.websocket, "state", None)

            st_name = getattr(st, "name", "")

            if st_name and st_name != "OPEN":

                return False

        return self.connected



    async def connect(self, _schedule_reconnect: bool = True) -> None:
        """Establish Deepgram WebSocket connection and start its receive loop."""
        if self._closing or self.is_open():
            return

        if not self.api_key:
            logger.info("DEEPGRAM_API_KEY not configured; operating in offline TTS mock mode.")
            self._mock_mode = True
            self.connected = True
            return

        async with self._connect_lock:
            if self._closing or self.is_open():
                return

            is_aura = self.model.startswith("aura-")
            endpoint_version = "v1" if is_aura else "v2"
            url = (
                f"wss://api.deepgram.com/{endpoint_version}/speak"
                f"?model={self.model}"
                f"&encoding={self.encoding}"
                f"&sample_rate={self.sample_rate}"
            )

            logger.info(
                "Connecting to Deepgram TTS (%s, %s, %s, %dHz)...",
                endpoint_version, self.model, self.encoding, self.sample_rate,
            )

            try:
                websocket = await websockets.connect(
                    url,
                    additional_headers={"Authorization": f"Token {self.api_key}"},
                    ping_interval=5,
                    ping_timeout=5,
                    max_size=None,
                )
                self.websocket = websocket
                self.connected = True
                self._mock_mode = False
                self._reconnect_attempts = 0
                logger.info("Deepgram TTS (%s) connected successfully.", self.model)
                self.receive_task = asyncio.create_task(self._receive_loop())
            except Exception as e:
                self.connected = False
                self.websocket = None
                self._mock_mode = False
                logger.warning("Deepgram connection failed: %s", e)
                if _schedule_reconnect and not self._closing:
                    self._start_reconnect()

    def _start_reconnect(self) -> None:
        """Start one bounded background reconnect loop."""
        if self._closing or not self.api_key:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Reconnect with exponential backoff after a Deepgram disconnect."""
        try:
            for attempt in range(1, self._max_reconnect_attempts + 1):
                if self._closing or self.is_open():
                    return
                delay = min(
                    self._reconnect_base_delay * (2 ** (attempt - 1)),
                    self._reconnect_max_delay,
                )
                self._reconnect_attempts = attempt
                logger.warning(
                    "Deepgram TTS disconnected; reconnect attempt %d/%d in %.1fs.",
                    attempt, self._max_reconnect_attempts, delay,
                )
                await asyncio.sleep(delay)
                if self._closing:
                    return
                await self.connect(_schedule_reconnect=False)
                if self.is_open():
                    logger.info("Deepgram TTS reconnected successfully on attempt %d.", attempt)
                    self._reconnect_attempts = 0
                    return
        except asyncio.CancelledError:
            return
        finally:
            if asyncio.current_task() is self._reconnect_task:
                self._reconnect_task = None

    async def _ensure_connected(self) -> bool:
        """Ensure a live Deepgram connection exists before sending commands."""
        if self._closing or not self.api_key:
            return self.is_open()
        if self.is_open():
            return True
        await self.connect()
        if self.is_open():
            return True
        self._start_reconnect()
        task = self._reconnect_task
        if task:
            try:
                await task
            except asyncio.CancelledError:
                return False
        return self.is_open()

    async def _receive_loop(self) -> None:

        """Listen for binary audio frames and control events from Deepgram."""

        websocket = None
        try:

            websocket = self.websocket
            if not websocket:

                return



            async for message in self.websocket:

                if isinstance(message, bytes):

                    now = time.perf_counter()



                    if not self.first_audio_received:

                        self.first_audio_received = True

                        if self.first_speak_sent_at:

                            latency = now - self.first_speak_sent_at

                            logger.info("DEEPGRAM FIRST AUDIO | Speak -> Audio: %.3fs", latency)



                    if not self.first_audio_event_reported:

                        self.first_audio_event_reported = True

                        if self.event_callback:

                            res = self.event_callback({

                                "type": "FirstAudio",

                                "timestamp": now,

                                "audio_bytes": len(message),

                            })

                            if asyncio.iscoroutine(res):

                                await res



                    if self.audio_callback:

                        res = self.audio_callback(message)

                        if asyncio.iscoroutine(res):

                            await res

                    continue



                try:

                    event = json.loads(message)

                except Exception:

                    continue



                event_type = event.get("type")

                if event_type in ["Flushed", "Metadata", "SpeechMetadata"]:

                    if self.event_callback:

                        res = self.event_callback({"type": "SpeechMetadata"})

                        if asyncio.iscoroutine(res):

                            await res

                elif event_type in ["Cleared", "SpeechInterrupted"]:

                    if self.event_callback:

                        res = self.event_callback({"type": "SpeechInterrupted"})

                        if asyncio.iscoroutine(res):

                            await res



        except asyncio.CancelledError:

            return

        except websockets.ConnectionClosed:

            pass

        except Exception as exc:

            logger.warning("Deepgram receive loop error: %s", exc)

        finally:
            self.connected = False
            if self.websocket is websocket:
                self.websocket = None
            if not self._closing and self.api_key and not self._mock_mode:
                self._start_reconnect()



    async def send_text(self, text: str) -> None:
        """Stream an incremental text chunk to Deepgram TTS engine."""
        if not text:
            return

        now = time.perf_counter()
        if self.first_speak_sent_at is None:
            self.turn_started_at = now
            self.first_speak_sent_at = now
            self.first_audio_received = False
            self.first_audio_event_reported = False

        if self._mock_mode:
            self._accumulated_text.append(text)
            return

        if not await self._ensure_connected():
            logger.warning("Deepgram TTS unavailable; dropping text after reconnect attempts.")
            return

        payload = {"type": "Speak", "text": text}
        try:
            await self.websocket.send(json.dumps(payload))
        except Exception as e:
            logger.warning("Error sending text to Deepgram: %s", e)
            self.connected = False
            self.websocket = None
            self._start_reconnect()
            if await self._ensure_connected():
                try:
                    await self.websocket.send(json.dumps(payload))
                except Exception as retry_error:
                    logger.warning("Retry sending text to Deepgram failed: %s", retry_error)
                    self.connected = False
                    self.websocket = None
                    self._start_reconnect()


    async def flush(self) -> None:
        """Signal end of utterance and flush audio synthesis."""
        if self._mock_mode:
            now = time.perf_counter()
            full_text = " ".join(self._accumulated_text)
            self._accumulated_text.clear()

            if self.event_callback:
                res = self.event_callback({"type": "FirstAudio", "timestamp": now, "audio_bytes": 1024})
                if asyncio.iscoroutine(res):
                    await res

            if self.audio_callback:
                import numpy as np
                duration = max(0.2, min(2.0, len(full_text) * 0.05))
                num_samples = int(self.sample_rate * duration)
                t = np.linspace(0, duration, num_samples, endpoint=False)
                sine = (np.sin(2 * np.pi * 440.0 * t) * 8000).astype(np.int16)
                res = self.audio_callback(sine.tobytes())
                if asyncio.iscoroutine(res):
                    await res

            if self.event_callback:
                res = self.event_callback({"type": "SpeechMetadata"})
                if asyncio.iscoroutine(res):
                    await res
            return

        if not await self._ensure_connected():
            logger.warning("Deepgram TTS unavailable; cannot flush audio.")
            return

        try:
            await self.websocket.send(json.dumps({"type": "Flush"}))
        except Exception as e:
            logger.warning("Error sending Flush to Deepgram: %s", e)
            self.connected = False
            self.websocket = None
            self._start_reconnect()


    async def interrupt(self) -> None:

        """Halt active synthesis and purge queued TTS frames (instant barge-in)."""

        self._accumulated_text.clear()

        if self._mock_mode:

            self.reset_turn()

            if self.event_callback:

                res = self.event_callback({"type": "SpeechInterrupted"})

                if asyncio.iscoroutine(res):

                    await res

            return



        if self.websocket and self.connected:

            cmd = "Clear" if self.model.startswith("aura-") else "Interrupt"

            try:

                await self.websocket.send(json.dumps({"type": cmd}))

                logger.info("Deepgram %s sent for barge-in interruption.", cmd)

            except Exception:

                pass



        self.reset_turn()



    def reset_turn(self) -> None:

        """Reset turn metrics and cancellation states."""

        self.turn_started_at = None

        self.first_speak_sent_at = None

        self.first_audio_received = False

        self.first_audio_event_reported = False

        self._accumulated_text.clear()



    async def close(self) -> None:
        """Close the WebSocket and stop all reconnect activity cleanly."""
        self._closing = True
        self.connected = False

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        self._reconnect_task = None

        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"type": "Close"}))
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None

        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
            self.receive_task = None

        self.reset_turn()


    def cancel(self) -> None:

        """Cancel synthesis immediately (TTSProvider conformity)."""

        self.cancel_current_synthesis()



    def cancel_current_synthesis(self) -> None:

        """Cancel synthesis immediately (TTSProvider interface)."""

        self._accumulated_text.clear()

        self.reset_turn()



    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> bytes:

        """Synthesize text into a complete audio frame buffer (TTSProvider interface)."""

        chunks: list[bytes] = []

        async for chunk in self.synthesize_stream(text):

            chunks.append(chunk)

        return b"".join(chunks)



    async def stream_synthesize(

        self,

        text_stream: AsyncIterator[str],

        voice_id: Optional[str] = None,

    ) -> AsyncIterator[bytes]:

        """Synthesize text chunks incrementally into streaming audio chunks without buffering delays."""

        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        old_audio_cb = self.audio_callback

        old_event_cb = self.event_callback



        async def on_audio(chunk: bytes):

            await queue.put(chunk)

            if old_audio_cb:

                res = old_audio_cb(chunk)

                if asyncio.iscoroutine(res):

                    await res



        async def on_event(event: dict):

            event_type = event.get("type")

            if event_type in ["SpeechMetadata", "Flushed", "Cleared", "SpeechInterrupted"]:

                await queue.put(None)

            if old_event_cb:

                res = old_event_cb(event)

                if asyncio.iscoroutine(res):

                    await res



        self.audio_callback = on_audio

        self.event_callback = on_event



        async def feed_text():

            try:

                async for text in text_stream:

                    await self.send_text(text)

                await self.flush()

            except Exception as exc:

                logger.warning("Error feeding text to Deepgram: %s", exc)

                await queue.put(None)



        feed_task = asyncio.create_task(feed_text())

        try:

            await self.connect()

            while True:

                chunk = await queue.get()

                if chunk is None:

                    break

                yield chunk

            await feed_task

        finally:

            self.audio_callback = old_audio_cb

            self.event_callback = old_event_cb

            if not feed_task.done():

                feed_task.cancel()



    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:

        """Streaming synthesis async generator yielding audio chunks immediately upon arrival."""

        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        old_audio_cb = self.audio_callback

        old_event_cb = self.event_callback



        async def on_audio(chunk: bytes):

            await queue.put(chunk)

            if old_audio_cb:

                res = old_audio_cb(chunk)

                if asyncio.iscoroutine(res):

                    await res



        async def on_event(event: dict):

            event_type = event.get("type")

            if event_type in ["SpeechMetadata", "Flushed", "Cleared", "SpeechInterrupted"]:

                await queue.put(None)

            if old_event_cb:

                res = old_event_cb(event)

                if asyncio.iscoroutine(res):

                    await res



        self.audio_callback = on_audio

        self.event_callback = on_event

        try:

            await self.connect()

            await self.send_text(text)

            await self.flush()

            while True:

                chunk = await queue.get()

                if chunk is None:

                    break

                yield chunk

        finally:

            self.audio_callback = old_audio_cb

            self.event_callback = old_event_cb





# Canonical Alias for backward compatibility

DeepgramTTS = DeepgramFluxTTS
