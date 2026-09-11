import numpy as np
import torch


class VADService:

    def __init__(
        self,
        sample_rate=16000,
        threshold=0.5
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold

        print("Loading Silero VAD...")

        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )

        self.model.eval()

        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks
        ) = utils

        # ==================================================
        # VAD CONFIGURATION
        # ==================================================
        self.vad_iterator = self.VADIterator(
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            # Configured to 350 ms silence threshold
            min_silence_duration_ms=350,
        )

        self.frame_size = 512
        self.frame_bytes = self.frame_size * 2

        self.buffer = bytearray()
        self.speech_buffer = bytearray()
        self.is_speaking = False

        print("Silero VAD loaded.")
        print("VAD threshold:", self.threshold)
        print("Minimum silence: 350 ms")

    def pcm16_to_float32(self, pcm_bytes):
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        return torch.from_numpy(audio)

    def process(self, pcm_bytes):
        result = {
            "speech_started": False,
            "speech_ended": False,
            "is_speech": self.is_speaking,
            "speech_audio": None
        }

        if not pcm_bytes:
            return result

        self.buffer.extend(pcm_bytes)

        while len(self.buffer) >= self.frame_bytes:
            frame = bytes(self.buffer[:self.frame_bytes])
            del self.buffer[:self.frame_bytes]

            audio = self.pcm16_to_float32(frame)

            try:
                vad_result = self.vad_iterator(audio)
            except Exception as e:
                print("VAD ERROR:", type(e).__name__, str(e))
                continue

            if vad_result and "start" in vad_result:
                self.is_speaking = True
                self.speech_buffer.extend(frame)
                result["speech_started"] = True
                result["is_speech"] = True
                print("🎙️ USER STARTED SPEAKING")

            elif self.is_speaking:
                self.speech_buffer.extend(frame)

            if vad_result and "end" in vad_result:
                self.is_speaking = False
                result["speech_ended"] = True
                result["is_speech"] = False
                result["speech_audio"] = bytes(self.speech_buffer)
                print("🛑 USER STOPPED SPEAKING")
                print("Speech segment:", len(self.speech_buffer), "bytes")
                self.speech_buffer.clear()

        return result

    def flush(self):
        if not self.speech_buffer:
            return None

        audio = bytes(self.speech_buffer)
        self.speech_buffer.clear()
        self.is_speaking = False
        self.buffer.clear()
        self.vad_iterator.reset_states()
        return audio

    def reset(self):
        self.buffer.clear()
        self.speech_buffer.clear()
        self.is_speaking = False
        self.vad_iterator.reset_states()