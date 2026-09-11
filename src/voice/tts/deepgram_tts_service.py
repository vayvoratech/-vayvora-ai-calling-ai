import os
import json
import time
import asyncio
import websockets


class DeepgramFluxTTS:

    def __init__(
        self,
        model=None,
        encoding=None,
        sample_rate=None
    ):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY environment variable is not set.")

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
        self.receive_task = None
        self.connected = False

        self.audio_callback = None
        self.event_callback = None

        self.turn_started_at = None
        self.first_speak_sent_at = None
        self.first_audio_received = False
        self.first_audio_event_reported = False

    async def connect(self):
        if self.connected:
            return

        is_aura = self.model.startswith("aura-")
        endpoint_version = "v1" if is_aura else "v2"

        url = (
            f"wss://api.deepgram.com/{endpoint_version}/speak"
            f"?model={self.model}"
            f"&encoding={self.encoding}"
            f"&sample_rate={self.sample_rate}"
        )

        print()
        print("================================")
        print("Connecting to Deepgram TTS")
        print("================================")
        print("Endpoint:", f"/{endpoint_version}/speak")
        print("Model:", self.model)
        print("Encoding:", self.encoding)
        print("Sample rate:", self.sample_rate)
        print()

        try:
            self.websocket = await websockets.connect(
                url,
                additional_headers={
                    "Authorization": f"Token {self.api_key}"
                },
                ping_interval=20,
                ping_timeout=20,
                max_size=None
            )
            self.connected = True
            print(f"Deepgram TTS ({self.model}) connected successfully.\n")

            self.receive_task = asyncio.create_task(
                self._receive_loop()
            )

        except Exception as e:
            print("DEEPGRAM CONNECTION ERROR:", type(e).__name__, str(e))
            self.connected = False
            self.websocket = None
            raise

    async def _receive_loop(self):
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    now = time.perf_counter()

                    if not self.first_audio_received:
                        self.first_audio_received = True
                        if self.first_speak_sent_at:
                            print(f"\nDEEPGRAM FIRST AUDIO | Speak → Audio: {now - self.first_speak_sent_at:.3f}s\n")

                    if not self.first_audio_event_reported:
                        self.first_audio_event_reported = True
                        if self.event_callback:
                            await self.event_callback({
                                "type": "FirstAudio",
                                "timestamp": now,
                                "audio_bytes": len(message),
                            })

                    if self.audio_callback:
                        await self.audio_callback(message)
                    continue

                try:
                    event = json.loads(message)
                except Exception:
                    continue

                event_type = event.get("type")

                if event_type in ["Flushed", "Metadata", "SpeechMetadata"]:
                    if self.event_callback:
                        await self.event_callback({"type": "SpeechMetadata"})

                elif event_type in ["Cleared", "SpeechInterrupted"]:
                    if self.event_callback:
                        await self.event_callback({"type": "SpeechInterrupted"})

        except asyncio.CancelledError:
            return
        except websockets.ConnectionClosed:
            pass
        finally:
            self.connected = False

    async def send_text(self, text: str):
        if not text or not self.connected:
            return

        now = time.perf_counter()
        if self.first_speak_sent_at is None:
            self.turn_started_at = now
            self.first_speak_sent_at = now
            self.first_audio_received = False
            self.first_audio_event_reported = False

        payload = {
            "type": "Speak",
            "text": text
        }
        await self.websocket.send(json.dumps(payload))
        print("TTS Speak:", repr(text))

    async def flush(self):
        if self.connected:
            await self.websocket.send(json.dumps({"type": "Flush"}))

    async def interrupt(self):
        if not self.connected:
            return

        cmd = "Clear" if self.model.startswith("aura-") else "Interrupt"
        try:
            await self.websocket.send(json.dumps({"type": cmd}))
            print(f"Deepgram {cmd} sent.")
        except Exception:
            pass

        self.reset_turn()

    def reset_turn(self):
        self.turn_started_at = None
        self.first_speak_sent_at = None
        self.first_audio_received = False
        self.first_audio_event_reported = False

    async def close(self):
        self.connected = False
        if self.websocket:
            try:
                cmd = "Close"
                await self.websocket.send(json.dumps({"type": cmd}))
                await self.websocket.close()
            except Exception:
                pass
        if self.receive_task:
            self.receive_task.cancel()
        self.reset_turn()