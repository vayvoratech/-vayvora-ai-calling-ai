"""Pipecat TTS adapter wrapping local TTSProvider implementations.

Provides real-time speech synthesis with barge-in interruption readiness,
generation validity checks, and latency telemetry.
"""

import time
from typing import AsyncIterator, Optional
from src.audio.schemas import TTSResult
from src.core.errors import BargeInInterruptionError, StaleGenerationError
from src.core.interfaces import TTSProvider
from src.logging import get_logger
from src.voice.context import CancellationToken

logger = get_logger("voice.adapters.tts")


class PipecatTTSAdapter:
    """Adapts TTSProvider into Pipecat real-time audio pipeline with cancellation support."""

    def __init__(self, provider: TTSProvider) -> None:
        self.provider = provider

    async def synthesize(
        self,
        text: str,
        generation_id: int,
        cancellation_token: CancellationToken,
        voice_id: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize text into speech audio ensuring generation remains valid."""
        # 1. Pre-synthesis cancellation guard
        if not cancellation_token.is_valid_generation(generation_id):
            logger.info(
                "TTS pre-synthesis aborted: generation %d was invalidated/cancelled.",
                generation_id,
            )
            raise StaleGenerationError(f"Generation {generation_id} is stale/cancelled.")

        if hasattr(self.provider, "reset_cancellation"):
            self.provider.reset_cancellation()

        t_start = time.perf_counter()
        try:
            result = await self.provider.synthesize(text, voice_id=voice_id)
        except Exception as exc:
            if cancellation_token.is_cancelled:
                raise BargeInInterruptionError(f"TTS synthesis interrupted: {exc}") from exc
            raise

        # 2. Post-synthesis cancellation guard (in case caller spoke during neural inference)
        if not cancellation_token.is_valid_generation(generation_id):
            logger.info(
                "TTS post-synthesis discarded: caller interrupted during generation %d.",
                generation_id,
            )
            raise StaleGenerationError(f"Generation {generation_id} invalidated post-synthesis.")

        latency = time.perf_counter() - t_start
        logger.debug(
            "TTS adapter synthesized %d bytes in %.3fs for generation %d",
            len(result.audio_data),
            latency,
            generation_id,
        )
        return result

    def cancel_active_synthesis(self) -> None:
        """Propagate barge-in cancellation to the underlying TTS engine."""
        if hasattr(self.provider, "cancel_current_synthesis"):
            self.provider.cancel_current_synthesis()
        logger.info("TTS adapter dispatched cancel_current_synthesis.")
