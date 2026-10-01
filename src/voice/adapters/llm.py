"""Pipecat ConversationEngine / LLM adapter.

Mediates between real-time voice transcripts and the core ConversationEngine,
enforcing generation cancellation checks without altering engine business logic,
RAG grounding, or verified MCP action contracts.
"""

import time
from typing import Optional
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.errors import BargeInInterruptionError, StaleGenerationError
from src.logging import get_logger
from src.state.models import ConversationState
from src.voice.context import CancellationToken

logger = get_logger("voice.adapters.llm")


class PipecatConversationAdapter:
    """Adapts ConversationEngine into Pipecat real-time turn pipeline."""

    def __init__(self, engine: ConversationEngine) -> None:
        self.engine = engine

    async def process_user_turn(
        self,
        state: ConversationState,
        user_text: str,
        generation_id: int,
        cancellation_token: CancellationToken,
    ) -> EngineTurnResult:
        """Execute a conversational intelligence turn through ConversationEngine."""
        # 1. Pre-turn cancellation guard
        if not cancellation_token.is_valid_generation(generation_id):
            logger.info(
                "Conversation turn aborted: generation %d was cancelled before engine execution.",
                generation_id,
            )
            raise StaleGenerationError(f"Generation {generation_id} stale before engine processing.")

        t_start = time.perf_counter()

        # 2. Invoke core ConversationEngine (intent, RAG, MCP, prompt synthesis, state mutation)
        turn_result = await self.engine.process_user_turn(state=state, user_message=user_text)

        t_elapsed = time.perf_counter() - t_start

        # 3. Post-turn cancellation guard (caller interrupted while engine was thinking/retrieving)
        if not cancellation_token.is_valid_generation(generation_id):
            logger.info(
                "Conversation turn completed in %.3fs but discarded: generation %d was invalidated.",
                t_elapsed,
                generation_id,
            )
            raise StaleGenerationError(f"Generation {generation_id} invalidated during engine execution.")

        logger.debug(
            "Conversation turn processed in %.3fs for generation %d. Response: '%s'",
            t_elapsed,
            generation_id,
            turn_result.response_text[:60] if turn_result.response_text else "",
        )
        return turn_result
