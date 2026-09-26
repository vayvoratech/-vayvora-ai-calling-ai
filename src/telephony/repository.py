"""Session persistence repositories for call summaries.

Provides MockCallSessionRepository for in-memory persistence and verification,
adhering to the CallSessionRepository contract.
"""

from typing import Dict, List, Optional

from src.core.errors import SessionPersistenceError
from src.logging import get_logger
from src.telephony.interfaces import CallSessionRepository
from src.telephony.models import CallSummary

logger = get_logger("telephony.repository")


class MockCallSessionRepository(CallSessionRepository):
    """In-memory call session summary repository for testing and validation."""

    def __init__(self, force_error: bool = False) -> None:
        self.records: Dict[str, CallSummary] = {}
        self.force_error = force_error

    async def persist_summary(self, summary: CallSummary) -> None:
        """Store CallSummary in memory."""
        if self.force_error:
            raise SessionPersistenceError("Simulated database failure persisting call summary.")

        self.records[summary.call_id] = summary
        logger.info(
            "Persisted CallSummary for call %s (Domain: %s, Actions: %d, Reason: %s)",
            summary.call_id,
            summary.domain,
            len(summary.completed_actions),
            summary.termination_reason,
        )

    async def get_summary(self, call_id: str) -> Optional[CallSummary]:
        """Fetch summary by call identifier."""
        return self.records.get(call_id)

    async def list_summaries(self, limit: int = 100) -> List[CallSummary]:
        """List stored summaries up to limit."""
        return list(self.records.values())[:limit]

    def clear(self) -> None:
        """Purge stored records."""
        self.records.clear()
