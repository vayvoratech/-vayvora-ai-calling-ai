"""Pipecat VAD adapter wrapping local VADProvider implementations.

Converts incoming raw audio frames into discrete speech segments, detects
speech boundaries with hysteresis (min speech / silence durations), and signals
barge-in onset to the pipeline.
"""

from typing import Optional
from src.audio.schemas import AudioChunk, VADResult
from src.core.interfaces import VADProvider
from src.logging import get_logger

logger = get_logger("voice.adapters.vad")


class PipecatVADAdapter:
    """Adapts VADProvider into Pipecat real-time audio pipeline."""

    def __init__(self, provider: VADProvider) -> None:
        self.provider = provider

    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """Process an AudioChunk through the underlying VAD provider."""
        if hasattr(self.provider, "process_chunk"):
            return self.provider.process_chunk(chunk)
        # Fallback to process_frame
        is_speech = self.provider.process_frame(chunk.data)
        return VADResult(
            is_speech=is_speech,
            timestamp=chunk.timestamp,
            duration=chunk.duration,
        )

    def reset(self) -> None:
        """Reset underlying VAD provider state machine."""
        if hasattr(self.provider, "reset"):
            self.provider.reset()
        logger.debug("VAD adapter reset.")
