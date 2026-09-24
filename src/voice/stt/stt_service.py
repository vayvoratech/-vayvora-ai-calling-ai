import io
import os
import time
import wave
from groq import Groq


class STTService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-large-v3-turbo",
        sample_rate: int = 16000,
        language: str = "en",
        prompt: str = "Vayvora Technology, AI, engineering services, consultation, appointment",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API Key not found. Please set the GROQ_API_KEY environment variable."
            )

        self.model = model
        self.sample_rate = sample_rate
        self.language = language
        self.prompt = prompt

        print()
        print("================================")
        print("Loading Groq Whisper STT")
        print("================================")
        print("Model:", self.model)
        print("Sample Rate:", f"{self.sample_rate}Hz")
        print("Language:", self.language)

        start = time.perf_counter()
        self.client = Groq(api_key=self.api_key)
        elapsed = time.perf_counter() - start

        print(f"Groq Client initialized in {elapsed:.3f}s")
        print("================================")
        print()

    def pcm16_to_wav_bytes(self, pcm_data: bytes) -> io.BytesIO:
        """Encapsulate raw PCM16 bytes into an in-memory WAV container."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)        # Mono
            wav_file.setsampwidth(2)       # 16-bit (2 bytes per sample)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_data)

        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"
        return wav_buffer

    def transcribe_pcm16(self, pcm_audio: bytes) -> str:
        if not pcm_audio:
            return ""

        print()
        print("--------------------------------")
        print("Starting Groq Whisper STT")
        print("--------------------------------")
        print("Audio bytes:", len(pcm_audio))

        # In-memory WAV packaging
        conversion_start = time.perf_counter()
        wav_file = self.pcm16_to_wav_bytes(pcm_audio)
        conversion_time = time.perf_counter() - conversion_start
        print(f"WAV packaging: {conversion_time:.4f}s")

        # Groq API Whisper Transcription
        start = time.perf_counter()
        try:
            transcription = self.client.audio.transcriptions.create(
                file=wav_file,
                model=self.model,
                language=self.language,
                prompt=self.prompt,  # Guides domain terminology (Vayvora, etc.)
                temperature=0.0,
                response_format="text",
            )
            text = str(transcription).strip()
        except Exception as e:
            print("Groq STT Error:", type(e).__name__, str(e))
            text = ""

        elapsed = time.perf_counter() - start

        print("Transcript:", text)
        print(f"Groq Latency: {elapsed:.3f}s")
        print("--------------------------------")
        print()

        return text