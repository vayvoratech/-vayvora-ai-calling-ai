"""Voice Activity Detection (VAD) provider implementations and abstractions.

Provides SileroVADProvider for local ONNX/PyTorch inference and MockVADProvider
for deterministic, fast unit tests without model downloads.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np

from src.audio.schemas import AudioChunk, VADResult
from src.config import Settings, get_settings
from src.core.errors import (
    AudioProviderInitializationError,
    InvalidAudioDataError,
    ModelUnavailableError,
    VADError,
    VADProcessingError,
)
from src.core.interfaces import VADProvider
from src.logging import get_logger

logger = get_logger("audio.vad")


class BaseVADProvider(VADProvider, ABC):
    """Abstract base class for VAD engines managing temporal speech/silence state transitions."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.5,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration

        # State machine tracking
        self.is_speaking: bool = False
        self._speech_accumulator: float = 0.0
        self._silence_accumulator: float = 0.0
        self._timeline_pos: float = 0.0

    @abstractmethod
    def compute_speech_probability(self, audio_bytes: bytes) -> float:
        """Compute speech probability in range [0.0, 1.0] for the raw audio frame."""
        pass

    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """Process an AudioChunk and return a stateful VADResult."""
        if not chunk.data or len(chunk.data) == 0:
            raise InvalidAudioDataError("Cannot process empty audio chunk in VAD.")

        if chunk.sample_rate != self.sample_rate:
            logger.warning(
                "Sample rate mismatch in VAD: expected %d Hz, received %d Hz.",
                self.sample_rate,
                chunk.sample_rate,
            )

        duration = chunk.duration
        if duration <= 0.0:
            bytes_per_sample = 2 * chunk.channels
            duration = len(chunk.data) / (chunk.sample_rate * bytes_per_sample)

        prob = self.compute_speech_probability(chunk.data)
        raw_speech = prob >= self.threshold

        speech_start = False
        speech_end = False

        if raw_speech:
            self._speech_accumulator += duration
            self._silence_accumulator = 0.0

            if not self.is_speaking and self._speech_accumulator >= self.min_speech_duration:
                self.is_speaking = True
                speech_start = True
                logger.debug("VAD: Speech start detected at %.2fs", self._timeline_pos)
        else:
            self._silence_accumulator += duration

            if self.is_speaking and self._silence_accumulator >= self.min_silence_duration:
                self.is_speaking = False
                speech_end = True
                self._speech_accumulator = 0.0
                logger.debug("VAD: Speech end detected at %.2fs", self._timeline_pos)
            elif not self.is_speaking:
                self._speech_accumulator = 0.0

        res = VADResult(
            is_speech=self.is_speaking or (raw_speech and self._speech_accumulator >= self.min_speech_duration),
            speech_start=speech_start,
            speech_end=speech_end,
            confidence=prob,
            timestamp=round(self._timeline_pos, 4),
            duration=round(duration, 4),
        )

        self._timeline_pos += duration
        return res

    def process_frame(self, audio_frame: bytes) -> bool:
        """Base interface implementation processing a single raw byte frame."""
        chunk = AudioChunk(
            data=audio_frame,
            sample_rate=self.sample_rate,
            channels=1,
            format="pcm16",
        )
        return self.process_chunk(chunk).is_speech

    def reset(self) -> None:
        """Reset internal speech detector and state machine accumulators."""
        self.is_speaking = False
        self._speech_accumulator = 0.0
        self._silence_accumulator = 0.0
        self._timeline_pos = 0.0
        logger.debug("VAD state machine reset.")


class MockVADProvider(BaseVADProvider):
    """Deterministic mock VAD provider for unit testing without model downloads."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.5,
        canned_probabilities: Optional[List[float]] = None,
        force_error: bool = False,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
        )
        self.canned_probabilities: List[float] = list(canned_probabilities or [])
        self.force_error: bool = force_error
        self.processed_chunks_count: int = 0

    def add_probability(self, prob: float) -> None:
        """Queue a canned speech probability value."""
        self.canned_probabilities.append(prob)

    def compute_speech_probability(self, audio_bytes: bytes) -> float:
        if self.force_error:
            raise VADProcessingError("Simulated VAD internal computation error.")

        self.processed_chunks_count += 1

        if self.canned_probabilities:
            return self.canned_probabilities.pop(0)

        # Energy-based heuristic: non-zero signal indicates speech
        try:
            pcm_data = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(pcm_data) == 0:
                return 0.0
            rms = np.sqrt(np.mean(pcm_data.astype(np.float32) ** 2))
            return 0.9 if rms > 100.0 else 0.05
        except Exception:
            return 0.0


class SileroVADProvider(BaseVADProvider):
    """Local Silero VAD implementation utilizing ONNX Runtime or PyTorch."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.5,
        model_path: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            settings=settings,
        )
        self.model_path = model_path
        self._session = None
        self._state = None
        self._init_model()

    def _init_model(self) -> None:
        """Load the Silero VAD ONNX model."""
        try:
            import onnxruntime as ort

            if self.model_path:
                self._session = ort.InferenceSession(self.model_path)

                # Silero VAD state shape: [2, batch, 128]
                import numpy as np
                self._state = np.zeros(
                    (2, 1, 128),
                    dtype=np.float32,
                )

                logger.info(
                    "Loaded Silero VAD ONNX model from %s",
                    self.model_path,
                )
            else:
                self._session = None

        except ImportError:
            self._session = None

    def compute_speech_probability(self, audio_bytes: bytes) -> float:
        if self._session is None:
            raise ModelUnavailableError(
                "Silero VAD ONNX model is not available or not loaded. "
                "Provide a valid model_path or use MockVADProvider for testing."
            )

        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            if len(audio_int16) == 0:
                return 0.0

            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            input_tensor = np.expand_dims(audio_float32, axis=0)

            # Silero ONNX inputs:
            # input -> [batch, samples]
            # state -> [2, batch, 128]
            # sr    -> scalar int64
            ort_inputs = {
                "input": input_tensor,
                "state": self._state,
                "sr": np.array(self.sample_rate, dtype=np.int64),
            }

            outputs = self._session.run(None, ort_inputs)

            # outputs:
            # outputs[0] -> speech probability
            # outputs[1] -> updated recurrent state
            probability = float(outputs[0][0][0])

            # Preserve state for the next audio chunk
            self._state = outputs[1]

            return probability

        except Exception as exc:
            raise VADProcessingError(
                f"Silero VAD inference failed: {exc}"
            ) from exc
    def reset(self) -> None:
        super().reset()

        self._state = np.zeros(
            (2, 1, 128),
            dtype=np.float32,
        )