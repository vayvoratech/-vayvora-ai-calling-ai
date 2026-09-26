"""Speech-to-Text (STT) provider implementations and abstractions.

Provides FasterWhisperSTTProvider for local whisper inference and MockSTTProvider
for fast deterministic testing without model downloads.
"""

from abc import ABC, abstractmethod
import io
import time
from typing import Any, AsyncIterator, List, Optional, Union
import numpy as np

from src.audio.schemas import AudioChunk, SpeechSegment, STTResult
from src.config import Settings, get_settings
from src.core.errors import (
    AudioProviderInitializationError,
    InvalidAudioDataError,
    ModelUnavailableError,
    STTError,
    STTTranscriptionError,
)
from src.core.interfaces import STTProvider
from src.logging import get_logger

logger = get_logger("audio.stt")


class BaseSTTProvider(STTProvider, ABC):
    """Abstract base class for STT engines converting audio buffers into STTResult models."""

    def __init__(
        self,
        model_name: str = "base.en",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        temperature: float = 0.0,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.temperature = temperature

    @abstractmethod
    async def transcribe(
        self,
        audio_data: Union[bytes, AudioChunk, SpeechSegment],
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribe speech audio into a structured STTResult model."""
        pass

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes],
        sample_rate: int = 16000,
    ) -> AsyncIterator[str]:
        """Incremental chunk transcription stream."""
        async for chunk in audio_stream:
            if chunk:
                res = await self.transcribe(chunk, sample_rate=sample_rate)
                yield res.text


class MockSTTProvider(BaseSTTProvider):
    """Deterministic mock STT provider for unit testing without downloading Whisper models."""

    def __init__(
        self,
        canned_transcriptions: Optional[List[str]] = None,
        default_text: str = "Hello, I am inquiring about courses.",
        confidence: float = 0.95,
        language: str = "en",
        force_error: bool = False,
    ) -> None:
        super().__init__()
        self.canned_transcriptions = list(canned_transcriptions or [])
        self.default_text = default_text
        self.confidence = confidence
        self.language = language
        self.force_error = force_error
        self.transcribed_calls: List[Dict[str, Any]] = []

    def add_transcription(self, text: str) -> None:
        """Queue a canned transcription result."""
        self.canned_transcriptions.append(text)

    async def transcribe(
        self,
        audio_data: Union[bytes, AudioChunk, SpeechSegment],
        sample_rate: int = 16000,
    ) -> STTResult:
        if self.force_error:
            raise STTTranscriptionError("Simulated STT transcription failure.")

        # Extract raw bytes and duration
        raw_bytes = b""
        duration = 1.0

        if isinstance(audio_data, SpeechSegment):
            raw_bytes = audio_data.audio_data
            duration = audio_data.duration
            sample_rate = audio_data.sample_rate
        elif isinstance(audio_data, AudioChunk):
            raw_bytes = audio_data.data
            duration = audio_data.duration
            sample_rate = audio_data.sample_rate
        elif isinstance(audio_data, bytes):
            raw_bytes = audio_data
            duration = max(0.1, len(raw_bytes) / (sample_rate * 2))
        else:
            raise InvalidAudioDataError(f"Unsupported audio type for STT: {type(audio_data)}")

        if not raw_bytes or len(raw_bytes) == 0:
            raise InvalidAudioDataError("Cannot transcribe empty audio buffer.")

        # Minimum audio frame check (at least 2 bytes for 1 PCM sample)
        if len(raw_bytes) < 2:
            raise InvalidAudioDataError("Audio buffer contains insufficient audio data.")

        t_start = time.perf_counter()

        text = self.canned_transcriptions.pop(0) if self.canned_transcriptions else self.default_text

        t_elapsed = max(0.001, time.perf_counter() - t_start)
        rtf = round(t_elapsed / max(0.001, duration), 4)

        result = STTResult(
            text=text,
            language=self.language,
            confidence=self.confidence,
            start_time=0.0,
            end_time=round(duration, 4),
            is_final=True,
            provider="mock-stt",
            model="mock-whisper",
            audio_duration=round(duration, 4),
            processing_time=round(t_elapsed, 4),
            real_time_factor=rtf,
        )

        self.transcribed_calls.append({
            "bytes_len": len(raw_bytes),
            "sample_rate": sample_rate,
            "result": result,
        })

        return result


class FasterWhisperSTTProvider(BaseSTTProvider):
    """Local faster-whisper STT provider using CT2-optimized Whisper models."""

    def __init__(
        self,
        model_name: str = "base.en",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        temperature: float = 0.0,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            language=language,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            temperature=temperature,
            settings=settings,
        )
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
        """Attempt to initialize the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Loaded faster-whisper model: %s on %s", self.model_name, self.device)
        except ImportError:
            self._model = None
        except Exception as exc:
            logger.warning("Could not load faster-whisper model: %s", exc)
            self._model = None

    async def transcribe(
        self,
        audio_data: Union[bytes, AudioChunk, SpeechSegment],
        sample_rate: int = 16000,
    ) -> STTResult:
        if self._model is None:
            raise ModelUnavailableError(
                f"faster-whisper model '{self.model_name}' is not installed or available. "
                "Install 'faster-whisper' or use MockSTTProvider."
            )

        # Extract bytes
        raw_bytes = (
            audio_data.audio_data
            if isinstance(audio_data, SpeechSegment)
            else (audio_data.data if isinstance(audio_data, AudioChunk) else audio_data)
        )

        if not raw_bytes or len(raw_bytes) == 0:
            raise InvalidAudioDataError("Cannot transcribe empty audio buffer.")

        t_start = time.perf_counter()
        try:
            # Convert PCM16 to float32 numpy array
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            duration = len(audio_np) / float(sample_rate)

            segments, info = self._model.transcribe(
                audio_np,
                beam_size=self.beam_size,
                language=self.language,
                temperature=self.temperature,
            )

            text_parts = [s.text.strip() for s in segments]
            full_text = " ".join(text_parts).strip()

            t_elapsed = time.perf_counter() - t_start
            rtf = round(t_elapsed / max(0.001, duration), 4)

            return STTResult(
                text=full_text,
                language=info.language if hasattr(info, "language") else self.language,
                confidence=None,
                start_time=0.0,
                end_time=round(duration, 4),
                is_final=True,
                provider="faster-whisper",
                model=self.model_name,
                audio_duration=round(duration, 4),
                processing_time=round(t_elapsed, 4),
                real_time_factor=rtf,
            )
        except Exception as exc:
            raise STTTranscriptionError(f"faster-whisper transcription failed: {exc}") from exc
