"""Text-to-Speech (TTS) provider implementations and abstractions.

Provides KokoroTTSProvider, PiperTTSProvider fallback, and MockTTSProvider
for deterministic, fast unit testing without model downloads.
"""

from abc import ABC, abstractmethod
import io
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import wave
import numpy as np

from src.audio.schemas import TTSResult
from src.config import Settings, get_settings
from src.core.errors import (
    AudioProviderInitializationError,
    InvalidAudioDataError,
    ModelUnavailableError,
    TTSError,
    TTSSynthesisError,
)
from src.core.interfaces import TTSProvider
from src.logging import get_logger

logger = get_logger("audio.tts")


def create_pcm_wav_bytes(
    duration: float = 1.0,
    sample_rate: int = 24000,
    frequency: float = 440.0,
) -> bytes:
    """Generate a clean, minimal valid WAV byte buffer using pure standard library wave."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(sample_rate)

        num_samples = int(sample_rate * max(0.1, duration))
        t = np.linspace(0, duration, num_samples, endpoint=False)
        audio = (np.sin(2 * np.pi * frequency * t) * 8000).astype(np.int16)
        w.writeframes(audio.tobytes())

    return buf.getvalue()


class BaseTTSProvider(TTSProvider, ABC):
    """Abstract base class for TTS synthesis engines."""

    def __init__(
        self,
        voice: str = "af_heart",
        language: str = "en-us",
        sample_rate: int = 24000,
        output_format: str = "wav",
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.voice = voice
        self.language = language
        self.sample_rate = sample_rate
        self.output_format = output_format
        self._is_cancelled: bool = False

    @abstractmethod
    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> TTSResult:
        """Synthesize text into a structured TTSResult containing audio bytes and metadata."""
        pass

    async def stream_synthesize(
        self, text_stream: AsyncIterator[str], voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """Synthesize incoming text chunks into audio stream."""
        async for chunk in text_stream:
            if self._is_cancelled:
                logger.info("TTS streaming synthesis interrupted.")
                break
            if chunk.strip():
                result = await self.synthesize(chunk, voice_id=voice_id)
                yield result.audio_data

    def cancel_current_synthesis(self) -> None:
        """Cancel ongoing synthesis to support caller interruption and low-latency barge-in."""
        self._is_cancelled = True
        logger.info("TTS: cancel_current_synthesis signal dispatched.")

    def reset_cancellation(self) -> None:
        """Reset the cancellation flag for subsequent synthesis turns."""
        self._is_cancelled = False


class MockTTSProvider(BaseTTSProvider):
    """Deterministic mock TTS provider for unit testing without downloading audio models."""

    def __init__(
        self,
        voice: str = "af_heart",
        language: str = "en-us",
        sample_rate: int = 24000,
        output_format: str = "wav",
        duration_per_word: float = 0.25,
        force_error: bool = False,
    ) -> None:
        super().__init__(
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            output_format=output_format,
        )
        self.duration_per_word = duration_per_word
        self.force_error = force_error
        self.synthesized_calls: List[Dict[str, Any]] = []

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> TTSResult:
        if self.force_error:
            raise TTSSynthesisError("Simulated TTS synthesis failure.")

        if not text or not text.strip():
            raise InvalidAudioDataError("Cannot synthesize empty text.")

        if self._is_cancelled:
            self._is_cancelled = False
            raise TTSSynthesisError("Synthesis was cancelled by interruption signal.")

        t_start = time.perf_counter()

        # Deterministic audio duration based on word count
        words = text.strip().split()
        duration = max(0.5, len(words) * self.duration_per_word)

        audio_bytes = create_pcm_wav_bytes(
            duration=duration,
            sample_rate=self.sample_rate,
            frequency=440.0,
        )

        t_elapsed = max(0.001, time.perf_counter() - t_start)
        rtf = round(t_elapsed / max(0.001, duration), 4)

        result = TTSResult(
            audio_data=audio_bytes,
            sample_rate=self.sample_rate,
            channels=1,
            format=self.output_format,
            text=text,
            duration=round(duration, 4),
            processing_time=round(t_elapsed, 4),
            real_time_factor=rtf,
            provider="mock-tts",
            voice=voice_id or self.voice,
        )

        self.synthesized_calls.append({
            "text": text,
            "duration": duration,
            "voice": voice_id or self.voice,
        })

        return result


class KokoroTTSProvider(BaseTTSProvider):
    """Local Kokoro TTS provider for natural, expressive neural speech synthesis."""

    def __init__(
        self,
        voice: str = "af_heart",
        language: str = "en-us",
        sample_rate: int = 24000,
        output_format: str = "wav",
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            output_format=output_format,
            settings=settings,
        )
        self._pipeline = None
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        """Attempt to load the Kokoro TTS pipeline."""
        try:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code="a")  # American English
            logger.info("Loaded Kokoro TTS pipeline successfully.")
        except ImportError:
            self._pipeline = None
        except Exception as exc:
            logger.warning("Could not initialize Kokoro TTS pipeline: %s", exc)
            self._pipeline = None

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> TTSResult:
        if self._pipeline is None:
            raise ModelUnavailableError(
                "Kokoro TTS is not installed or available. "
                "Install 'kokoro' package or use MockTTSProvider for testing."
            )

        if not text or not text.strip():
            raise InvalidAudioDataError("Cannot synthesize empty text.")

        t_start = time.perf_counter()
        try:
            voice_to_use = voice_id or self.voice
            generator = self._pipeline(text, voice=voice_to_use, speed=1.0, split_pattern=r"\n+")

            audio_segments = []
            for _, _, audio in generator:
                if self._is_cancelled:
                    raise TTSSynthesisError("Synthesis cancelled by interruption.")
                audio_segments.append(audio)

            if not audio_segments:
                raise TTSSynthesisError("Kokoro produced empty audio segments.")

            full_audio = np.concatenate(audio_segments)
            duration = len(full_audio) / float(self.sample_rate)

            # Convert to 16-bit PCM WAV
            pcm_bytes = (full_audio * 32767.0).astype(np.int16).tobytes()

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes(pcm_bytes)
            wav_bytes = buf.getvalue()

            t_elapsed = time.perf_counter() - t_start
            rtf = round(t_elapsed / max(0.001, duration), 4)

            return TTSResult(
                audio_data=wav_bytes,
                sample_rate=self.sample_rate,
                channels=1,
                format=self.output_format,
                text=text,
                duration=round(duration, 4),
                processing_time=round(t_elapsed, 4),
                real_time_factor=rtf,
                provider="kokoro",
                voice=voice_to_use,
            )
        except Exception as exc:
            raise TTSSynthesisError(f"Kokoro synthesis failed: {exc}") from exc


class PiperTTSProvider(BaseTTSProvider):
    """Local Piper TTS provider offering fast, low-latency ONNX speech synthesis."""

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        language: str = "en-us",
        sample_rate: int = 22050,
        output_format: str = "wav",
        model_path: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            output_format=output_format,
            settings=settings,
        )
        self.model_path = model_path
        self._voice_model = None

    async def synthesize(self, text: str, voice_id: Optional[str] = None) -> TTSResult:
        if self._voice_model is None:
            raise ModelUnavailableError(
                "Piper TTS model is not installed or available. "
                "Provide a valid model_path or use MockTTSProvider for testing."
            )

        if not text or not text.strip():
            raise InvalidAudioDataError("Cannot synthesize empty text.")

        raise NotImplementedError("Piper execution requires local binary / ONNX model.")
