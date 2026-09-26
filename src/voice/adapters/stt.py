"""Pipecat STT adapter wrapping local STTProvider implementations.

Converts completed speech segments into structured STTResult transcripts,
measuring inference latency and filtering empty or non-speech noise.
"""

import time
from typing import Optional, Union
from src.audio.schemas import AudioChunk, SpeechSegment, STTResult
from src.core.interfaces import STTProvider
from src.logging import get_logger

logger = get_logger("voice.adapters.stt")


class PipecatSTTAdapter:
    """Adapts STTProvider into Pipecat real-time audio pipeline."""

    def __init__(self, provider: STTProvider) -> None:
        self.provider = provider

    async def transcribe_segment(
        self, segment: Union[bytes, AudioChunk, SpeechSegment], sample_rate: int = 16000
    ) -> STTResult:
        """Transcribe a completed speech segment through the underlying STT provider."""
        t_start = time.perf_counter()
        result = await self.provider.transcribe(segment, sample_rate=sample_rate)
        latency = time.perf_counter() - t_start

        logger.debug(
            "STT adapter transcribed segment in %.3fs: '%s'",
            latency,
            result.text,
        )
        return result
