"""Session persistence repositories for call summaries.

Provides MockCallSessionRepository for in-memory persistence and verification,
adhering to the CallSessionRepository contract.
"""

from typing import Any, Dict, List, Optional

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
        self.interactions: List[Dict[str, Any]] = []

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

    async def save_call_interaction(
        self,
        session_id: str,
        speaker: str,
        text: str,
    ) -> Optional[int]:
        """Store interaction turn in memory."""
        if self.force_error:
            raise SessionPersistenceError("Simulated database failure persisting call interaction.")
        if not hasattr(self, "interactions"):
            self.interactions = []
        rec_id = len(self.interactions) + 1
        self.interactions.append({
            "id": rec_id,
            "session_id": session_id,
            "speaker": speaker,
            "text": text,
        })
        return rec_id

    async def get_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve stored interaction turns for a session."""
        if not hasattr(self, "interactions"):
            return []
        return [i for i in self.interactions if i.get("session_id") == session_id]

    async def get_summary(self, call_id: str) -> Optional[CallSummary]:
        """Fetch summary by call identifier."""
        return self.records.get(call_id)

    async def list_summaries(self, limit: int = 100) -> List[CallSummary]:
        """List stored summaries up to limit."""
        return list(self.records.values())[:limit]

    def clear(self) -> None:
        """Purge stored records."""
        self.records.clear()
        if hasattr(self, "interactions"):
            self.interactions.clear()


class PostgresCallSessionRepository(CallSessionRepository):
    """PostgreSQL call session repository persisting call summaries and interactions."""

    def __init__(
        self,
        postgres_repo: Optional[Any] = None,
        fallback_repo: Optional[CallSessionRepository] = None,
    ) -> None:
        if postgres_repo is None:
            try:
                from src.database.repository import PostgresRepository
                self.postgres_repo = PostgresRepository()
            except Exception:
                self.postgres_repo = None
        else:
            self.postgres_repo = postgres_repo

        self.fallback = fallback_repo or MockCallSessionRepository()

    async def save_call_interaction(
        self,
        session_id: str,
        speaker: str,
        text: str,
    ) -> Optional[int]:
        """Store interaction turn in fallback and PostgreSQL."""
        if hasattr(self.fallback, "save_call_interaction"):
            await self.fallback.save_call_interaction(session_id, speaker, text)

        if self.postgres_repo is not None:
            try:
                return await self.postgres_repo.save_call_interaction(
                    session_id=session_id,
                    speaker=speaker,
                    text=text,
                )
            except Exception as exc:
                logger.warning("Failed to save call interaction to PostgreSQL: %s", type(exc).__name__)
                return None
        return None

    async def get_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve stored interaction turns for a session."""
        if self.postgres_repo is not None:
            try:
                res = await self.postgres_repo.get_call_interactions(session_id)
                if res:
                    return res
            except Exception as exc:
                logger.warning("Failed to get call interactions from PostgreSQL: %s", type(exc).__name__)
        if hasattr(self.fallback, "get_interactions"):
            return await self.fallback.get_interactions(session_id)
        return []

    async def persist_summary(self, summary: CallSummary) -> None:
        """Store CallSummary in memory and persist caller and interaction into PostgreSQL."""
        await self.fallback.persist_summary(summary)

        if self.postgres_repo is not None:
            try:
                # 1. Persist caller as lead ONLY if valid phone number is available
                clean_phone = None
                if summary.caller and summary.caller.get("phone"):
                    from src.database.repository import normalize_lead_phone
                    clean_phone = normalize_lead_phone(summary.caller.get("phone"))

                if clean_phone:
                    await self.postgres_repo.upsert_lead(
                        name=summary.caller.get("name"),
                        phone_number=clean_phone,
                        email=summary.caller.get("email"),
                        status=summary.business_status or "completed",
                        interested_in=summary.important_entities.get("course")
                        or summary.important_entities.get("domain")
                        or summary.domain,
                        metadata={
                            "call_id": summary.call_id,
                            "session_id": summary.session_id,
                            "duration": summary.duration,
                            "termination_reason": summary.termination_reason,
                            "completed_actions": summary.completed_actions,
                        },
                    )

                # 2. Persist dialogue/summary interaction to call_interactions
                await self.postgres_repo.save_call_interaction(
                    session_id=summary.session_id,
                    speaker="system",
                    text=f"Call completed. Domain: {summary.domain}, Intent: {summary.primary_intent}, Duration: {summary.duration}s, Reason: {summary.termination_reason}",
                )
                logger.info("Persisted CallSummary for call %s to PostgreSQL", summary.call_id)
            except Exception as exc:
                logger.warning("Failed to persist call summary to PostgreSQL (fallback preserved): %s", exc)

    async def get_summary(self, call_id: str) -> Optional[CallSummary]:
        """Fetch summary by call identifier."""
        return await self.fallback.get_summary(call_id)

    async def list_summaries(self, limit: int = 100) -> List[CallSummary]:
        """List stored summaries up to limit."""
        return await self.fallback.list_summaries(limit)

    def clear(self) -> None:
        """Purge stored records in fallback."""
        if hasattr(self.fallback, "clear"):
            self.fallback.clear()

