"""PostgreSQL repositories for voice call interactions, leads, and HR queues.

Interacts with the verified PostgreSQL schema (leads, call_interactions, hr_followup_queue)
with non-terminating error recovery, keeping the real-time voice pipeline resilient.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from src.database.connection import DatabasePool, get_db_pool
from src.logging import get_logger

logger = get_logger("database.repository")


class PostgresRepository:
    """Async repository executing operations against PostgreSQL tables."""

    def __init__(self, pool: Optional[DatabasePool] = None) -> None:
        self.pool = pool or get_db_pool()

    # -------------------------------------------------------------------------
    # 1. Call Interactions (Dialogue Transcript Turns)
    # -------------------------------------------------------------------------

    async def save_call_interaction(
        self,
        session_id: str,
        speaker: str,
        text: str,
    ) -> Optional[int]:
        """Record a single dialogue turn in call_interactions table."""
        try:
            pool = await self.pool.get_pool()
            query = """
                INSERT INTO call_interactions (session_id, speaker, text, created_at)
                VALUES ($1, $2, $3, $4)
                RETURNING id;
            """
            now = datetime.now(timezone.utc)
            eff_session = session_id or "session_default"
            eff_speaker = speaker or "system"
            eff_text = text or ""
            async with pool.acquire() as conn:
                record_id = await conn.fetchval(query, eff_session, eff_speaker, eff_text, now)
                logger.debug("Persisted interaction turn %s for session %s", record_id, eff_session)
                return record_id
        except Exception as exc:
            logger.warning("Failed to persist call interaction to PostgreSQL: %s", exc)
            return None

    async def get_call_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve chronological interactions for a given call session."""
        try:
            pool = await self.pool.get_pool()
            query = """
                SELECT id, session_id, speaker, text, created_at
                FROM call_interactions
                WHERE session_id = $1
                ORDER BY created_at ASC;
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, session_id)
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to fetch call interactions for %s: %s", session_id, exc)
            return []

    # -------------------------------------------------------------------------
    # 2. Leads (CRM Lead Updates and Creations)
    # -------------------------------------------------------------------------

    async def upsert_lead(
        self,
        name: Optional[str] = None,
        phone_number: Optional[str] = None,
        email: Optional[str] = None,
        status: Optional[str] = "new",
        interested_in: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Insert or update lead information in the leads table."""
        try:
            pool = await self.pool.get_pool()
            meta_json = json.dumps(metadata or {})
            now = datetime.now(timezone.utc)
            eff_name = name or "Lead Prospect"
            eff_status = status or "new"

            # Check if lead exists by phone or email
            async with pool.acquire() as conn:
                existing_id: Optional[int] = None
                if phone_number:
                    existing_id = await conn.fetchval(
                        "SELECT id FROM leads WHERE phone_number = $1 LIMIT 1;", phone_number
                    )
                elif email:
                    existing_id = await conn.fetchval(
                        "SELECT id FROM leads WHERE email = $1 LIMIT 1;", email
                    )

                if existing_id:
                    update_query = """
                        UPDATE leads
                        SET name = COALESCE($1, name),
                            email = COALESCE($2, email),
                            status = COALESCE($3, status),
                            interested_in = COALESCE($4, interested_in),
                            metadata = leads.metadata || $5::jsonb,
                            updated_at = $6
                        WHERE id = $7
                        RETURNING id;
                    """
                    lead_id = await conn.fetchval(
                        update_query, name, email, eff_status, interested_in, meta_json, now, existing_id
                    )
                    logger.info("Updated existing lead ID %s in PostgreSQL", lead_id)
                    return lead_id
                else:
                    eff_phone = phone_number or f"+1555{int(now.timestamp()) % 10000000:07d}"
                    insert_query = """
                        INSERT INTO leads (name, phone_number, email, status, interested_in, metadata, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
                        RETURNING id;
                    """
                    lead_id = await conn.fetchval(
                        insert_query, eff_name, eff_phone, email, eff_status, interested_in, meta_json, now, now
                    )
                    logger.info("Created new lead ID %s in PostgreSQL", lead_id)
                    return lead_id
        except Exception as exc:
            logger.warning("Failed to upsert lead to PostgreSQL: %s", exc)
            return None

    async def list_leads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List stored leads."""
        try:
            pool = await self.pool.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM leads ORDER BY updated_at DESC LIMIT $1;", limit)
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to list leads: %s", exc)
            return []

    # -------------------------------------------------------------------------
    # 3. HR Followup Queue
    # -------------------------------------------------------------------------

    async def create_hr_followup(
        self,
        candidate_name: str,
        session_id: Optional[str] = None,
        phone_number: Optional[str] = None,
        email: Optional[str] = None,
        reason: Optional[str] = None,
        status: str = "queued",
    ) -> Optional[int]:
        """Queue a candidate callback in hr_followup_queue table."""
        try:
            pool = await self.pool.get_pool()
            query = """
                INSERT INTO hr_followup_queue (session_id, name, phone_number, email, reason, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id;
            """
            now = datetime.now(timezone.utc)
            eff_session = session_id or "session_default"
            eff_candidate = candidate_name or "Applicant"
            eff_phone = phone_number or "+10000000000"
            eff_reason = reason or "HR callback requested"
            eff_status = status or "queued"

            async with pool.acquire() as conn:
                ticket_id = await conn.fetchval(
                    query, eff_session, eff_candidate, eff_phone, email, eff_reason, eff_status, now
                )
                logger.info("Created HR follow-up ticket %s in PostgreSQL for %s", ticket_id, eff_candidate)
                return ticket_id
        except Exception as exc:
            logger.warning("Failed to queue HR follow-up to PostgreSQL: %s", exc)
            return None

    async def list_hr_followups(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List queued HR follow-up callbacks."""
        try:
            pool = await self.pool.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM hr_followup_queue ORDER BY created_at DESC LIMIT $1;", limit)
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to list HR followups: %s", exc)
            return []
