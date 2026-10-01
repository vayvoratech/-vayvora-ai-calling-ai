"""Local Audio Pipeline Utility connecting AudioChunk -> VAD -> SpeechSegment -> STT.

Provides deterministic audio framing and speech segmentation strictly for testing
and validating audio provider implementations without owning conversation state,
RAG, MCP, or telephony orchestration.
"""

from typing import List, Optional, Tuple
from src.audio.schemas import AudioChunk, SpeechSegment, STTResult, VADResult
from src.audio.stt import BaseSTTProvider
from src.audio.vad import BaseVADProvider
from src.core.errors import AudioPipelineError
from src.logging import get_logger

logger = get_logger("audio.pipeline")


class AudioPipeline:
    """Lightweight test pipeline connecting VAD segmentation to STT transcription."""

    def __init__(
        self,
        vad_provider: BaseVADProvider,
        stt_provider: BaseSTTProvider,
    ) -> None:
        self.vad = vad_provider
        self.stt = stt_provider
        self._speech_chunks: List[AudioChunk] = []
        self._speech_active: bool = False
        self._speech_start_time: float = 0.0

    async def process_chunk(
        self, chunk: AudioChunk
    ) -> Tuple[VADResult, Optional[STTResult]]:
        """Process an AudioChunk through VAD and transcribe upon speech termination."""
        try:
            vad_res = self.vad.process_chunk(chunk)
        except Exception as exc:
            logger.error("VAD chunk evaluation failed: %s", exc)
            raise AudioPipelineError(f"AudioPipeline VAD failure: {exc}") from exc

        stt_res: Optional[STTResult] = None

        if vad_res.speech_start:
            self._speech_active = True
            self._speech_start_time = vad_res.timestamp
            self._speech_chunks = [chunk]
            logger.debug("Pipeline: Speech started at %.2fs", vad_res.timestamp)

        elif self._speech_active:
            self._speech_chunks.append(chunk)

            if vad_res.speech_end:
                logger.debug(
                    "Pipeline: Speech ended. Emitting segment with %d chunks.",
                    len(self._speech_chunks),
                )
                segment = self._build_speech_segment(end_time=vad_res.timestamp)
                self._speech_active = False
                self._speech_chunks = []

                try:
                    stt_res = await self.stt.transcribe(segment)
                except Exception as exc:
                    logger.error("STT transcription failed in pipeline: %s", exc)
                    raise AudioPipelineError(f"AudioPipeline STT failure: {exc}") from exc

        return vad_res, stt_res

    def _build_speech_segment(self, end_time: float) -> SpeechSegment:
        """Concatenate accumulated speech audio chunks into a single SpeechSegment."""
        if not self._speech_chunks:
            raise AudioPipelineError("Cannot assemble speech segment from empty chunks.")

        sample_rate = self._speech_chunks[0].sample_rate
        channels = self._speech_chunks[0].channels
        audio_data = b"".join(c.data for c in self._speech_chunks)
        duration = sum(c.duration for c in self._speech_chunks)

        return SpeechSegment(
            audio_data=audio_data,
            sample_rate=sample_rate,
            channels=channels,
            start_time=self._speech_start_time,
            end_time=end_time,
            duration=round(duration, 4),
            chunks_count=len(self._speech_chunks),
        )

    async def flush(self) -> Optional[STTResult]:
        """Force flush accumulated speech chunks into STT (e.g., at end of session)."""
        if self._speech_active and self._speech_chunks:
            segment = self._build_speech_segment(
                end_time=self._speech_start_time + sum(c.duration for c in self._speech_chunks)
            )
            self._speech_active = False
            self._speech_chunks = []
            return await self.stt.transcribe(segment)
        return None

    def reset(self) -> None:
        """Reset internal VAD and accumulator buffers."""
        self.vad.reset()
        self._speech_chunks.clear()
        self._speech_active = False
        self._speech_start_time = 0.0
        logger.debug("Audio pipeline reset.")
