"""PostgreSQL repositories for voice call interactions, leads, and HR queues.

Interacts with the verified PostgreSQL schema (leads, call_interactions, hr_followup_queue)
using atomic operations, non-terminating error recovery, and strict privacy safeguards.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from src.database.connection import DatabasePool, get_db_pool
from src.logging import get_logger

logger = get_logger("database.repository")

PROHIBITED_PHONE_PLACEHOLDERS = {
    "default caller",
    "john doe",
    "jane doe",
    "unknown",
    "none",
    "null",
    "n/a",
    "undefined",
}


def mask_phone_number(phone: Optional[str]) -> str:
    """Return a masked representation of a phone number for safe logging (e.g., '***-1234')."""
    if not phone:
        return "<none>"
    digits = re.sub(r"[^\d]", "", str(phone))
    if len(digits) <= 4:
        return "***"
    return f"***-{digits[-4:]}"


def mask_email(email: Optional[str]) -> str:
    """Return a masked representation of an email for safe logging (e.g., 'j***e@example.com')."""
    if not email:
        return "<none>"
    val = str(email).strip()
    if "@" not in val:
        return "***"
    parts = val.split("@", 1)
    user, domain = parts[0], parts[1]
    if len(user) <= 1:
        masked_user = "*"
    elif len(user) == 2:
        masked_user = f"{user[0]}*"
    else:
        masked_user = f"{user[0]}***{user[-1]}"
    return f"{masked_user}@{domain}"


def mask_sensitive_text(text: Optional[str]) -> str:
    """Mask phone numbers and email addresses embedded within conversational text."""
    if not text:
        return ""
    val = str(text)

    # 1. Mask email patterns
    def _sub_email(m: re.Match[str]) -> str:
        return mask_email(m.group(0))

    val = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", _sub_email, val)

    # 2. Mask phone number patterns (7 to 16 digits, with optional '+' or formatting symbols)
    def _sub_phone(m: re.Match[str]) -> str:
        raw_digits = re.sub(r"[^\d]", "", m.group(0))
        if 7 <= len(raw_digits) <= 16:
            return mask_phone_number(raw_digits)
        return m.group(0)

    val = re.sub(r"(?:\+?\d[\d\s\-\(\)\.]{5,}\d)", _sub_phone, val)
    return val


def normalize_lead_phone(phone: Optional[str]) -> Optional[str]:
    """Normalize a phone number for repository operations without fabricating numbers or country codes.

    Rules:
    - Rejects empty, whitespace-only, or prohibited placeholder strings.
    - Strips spaces, dashes, parentheses, and dots.
    - Retains leading '+' if present on international inputs.
    - Rejects strings with fewer than 7 digits or more than 16 digits.
    - Rejects dummy numbers (all identical digits, e.g. '0000000000', '9999999999').
    - Rejects sequential dummy numbers (e.g. '1234567890', '9876543210').
    """
    if not phone or not isinstance(phone, str):
        return None
    val = phone.strip()
    if not val or val.lower() in PROHIBITED_PHONE_PLACEHOLDERS:
        return None

    has_plus = val.startswith("+")
    digits = re.sub(r"[^\d]", "", val)
    if len(digits) < 7 or len(digits) > 16:
        return None

    # Check for identical repeating digits (dummy/placeholder)
    if len(set(digits)) <= 1:
        return None

    # Common sequential placeholders
    core = digits[-10:] if len(digits) >= 10 else digits
    if core in ("1234567890", "0123456789", "9876543210", "0987654321"):
        return None

    return f"+{digits}" if has_plus else digits


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
            logger.warning("Failed to persist call interaction to PostgreSQL: %s", type(exc).__name__)
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
            logger.warning("Failed to fetch call interactions for %s: %s", session_id, type(exc).__name__)
            return []

    async def get_latest_interaction_summary(self, session_id: str) -> Optional[str]:
        """Retrieve the latest summary interaction text for a specific session."""
        if not session_id:
            return None
        try:
            pool = await self.pool.get_pool()
            query = """
                SELECT text
                FROM call_interactions
                WHERE session_id = $1 AND speaker = 'system'
                ORDER BY created_at DESC
                LIMIT 1;
            """
            async with pool.acquire() as conn:
                return await conn.fetchval(query, session_id)
        except Exception as exc:
            logger.warning("Failed to fetch interaction summary for %s: %s", session_id, type(exc).__name__)
            return None

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
        """Insert or update lead information in the leads table atomically.

        Uses PostgreSQL INSERT ... ON CONFLICT (phone_number) DO UPDATE to eliminate
        race conditions. Rejects missing or invalid phone numbers without fabricating
        dummy numbers. Preserves existing non-null details when updating.
        """
        clean_phone = normalize_lead_phone(phone_number)
        if not clean_phone:
            logger.info("Skipping lead upsert: valid phone number is required but missing or invalid.")
            return None

        try:
            pool = await self.pool.get_pool()
            meta_json = json.dumps(metadata or {})
            now = datetime.now(timezone.utc)
            eff_name = name.strip() if name and name.strip() else None
            eff_email = email.strip() if email and email.strip() else None
            eff_status = status.strip() if status and status.strip() else "new"
            eff_interest = interested_in.strip() if interested_in and interested_in.strip() else None

            # Parameterized atomic upsert on verified UNIQUE (phone_number) constraint
            query = """
                INSERT INTO leads (name, phone_number, email, status, interested_in, metadata, created_at, updated_at)
                VALUES (COALESCE($1, 'Lead Prospect'), $2, $3, COALESCE($4, 'new'), $5, $6::jsonb, $7, $8)
                ON CONFLICT (phone_number) DO UPDATE
                SET name = COALESCE($1, leads.name),
                    email = COALESCE($3, leads.email),
                    status = COALESCE($4, leads.status),
                    interested_in = COALESCE($5, leads.interested_in),
                    metadata = leads.metadata || $6::jsonb,
                    updated_at = $8
                RETURNING id;
            """
            async with pool.acquire() as conn:
                lead_id = await conn.fetchval(
                    query,
                    eff_name,
                    clean_phone,
                    eff_email,
                    eff_status,
                    eff_interest,
                    meta_json,
                    now,
                    now,
                )
                logger.info(
                    "Atomic upsert completed for lead ID %s (phone: %s)",
                    lead_id,
                    mask_phone_number(clean_phone),
                )
                return lead_id
        except Exception as exc:
            logger.warning("Failed to upsert lead to PostgreSQL: %s", type(exc).__name__)
            return None

    async def get_lead_by_phone(self, phone_number: Optional[str]) -> Optional[Dict[str, Any]]:
        """Retrieve an existing lead record by normalized phone number using parameterized SQL.

        Returns:
            Dict representing lead row (with parsed JSON metadata), or None if not found or invalid phone.
        """
        clean_phone = normalize_lead_phone(phone_number)
        if not clean_phone:
            logger.debug("get_lead_by_phone skipped: invalid or missing phone number.")
            return None

        try:
            pool = await self.pool.get_pool()
            query = """
                SELECT id, name, phone_number, email, status, interested_in, metadata, created_at, updated_at
                FROM leads
                WHERE phone_number = $1
                LIMIT 1;
            """
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, clean_phone)
                if not row:
                    logger.debug("No lead found for phone %s", mask_phone_number(clean_phone))
                    return None

                lead_dict = dict(row)
                if isinstance(lead_dict.get("metadata"), str):
                    try:
                        lead_dict["metadata"] = json.loads(lead_dict["metadata"])
                    except Exception:
                        pass
                logger.debug("Found lead ID %s for phone %s", lead_dict.get("id"), mask_phone_number(clean_phone))
                return lead_dict
        except Exception as exc:
            logger.warning("Failed to fetch lead by phone from PostgreSQL: %s", type(exc).__name__)
            return None

    async def list_leads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List stored leads."""
        try:
            pool = await self.pool.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM leads ORDER BY updated_at DESC LIMIT $1;", limit)
                results: List[Dict[str, Any]] = []
                for r in rows:
                    d = dict(r)
                    if isinstance(d.get("metadata"), str):
                        try:
                            d["metadata"] = json.loads(d["metadata"])
                        except Exception:
                            pass
                    results.append(d)
                return results
        except Exception as exc:
            logger.warning("Failed to list leads: %s", type(exc).__name__)
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
                logger.info(
                    "Created HR follow-up ticket %s in PostgreSQL for candidate (phone: %s)",
                    ticket_id,
                    mask_phone_number(eff_phone),
                )
                return ticket_id
        except Exception as exc:
            logger.warning("Failed to queue HR follow-up to PostgreSQL: %s", type(exc).__name__)
            return None

    async def list_hr_followups(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List queued HR follow-up callbacks."""
        try:
            pool = await self.pool.get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM hr_followup_queue ORDER BY created_at DESC LIMIT $1;", limit)
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to list HR followups: %s", type(exc).__name__)
            return []

    # -------------------------------------------------------------------------
    # 4. Appointments & Site Visits
    # -------------------------------------------------------------------------

    async def create_appointment(
        self,
        phone_number: str,
        title: str,
        event_date: str,
        event_time: str,
        slot: str,
        project_name: Optional[str] = None,
        external_event_id: Optional[str] = None,
        lead_id: Optional[int] = None,
        description: Optional[str] = None,
        attendee_name: Optional[str] = None,
        attendee_email: Optional[str] = None,
        timezone_str: str = "Asia/Kolkata",
        status: str = "confirmed",
    ) -> Optional[Dict[str, Any]]:
        """Persist a confirmed appointment in the appointments table.

        Associates the record with the verified lead when available, and enforces
        strict phone normalization without fabricating synthetic identities.
        """
        clean_phone = normalize_lead_phone(phone_number)
        if not clean_phone:
            logger.warning("Skipping appointment creation: valid phone number is required but missing or invalid.")
            return None

        # Link to existing lead if not explicitly provided
        effective_lead_id = lead_id
        if effective_lead_id is None:
            try:
                existing_lead = await self.get_lead_by_phone(clean_phone)
                if existing_lead:
                    effective_lead_id = existing_lead.get("id")
            except Exception as lead_err:
                logger.debug("Could not link lead during appointment creation: %s", lead_err)

        try:
            pool = await self.pool.get_pool()
            now = datetime.now(timezone.utc)
            eff_title = (title or "Property Site Visit").strip()
            eff_date = (event_date or "").strip()
            eff_time = (event_time or "").strip()
            eff_slot = (slot or f"{eff_date} {eff_time}").strip()
            eff_project = project_name.strip() if project_name and project_name.strip() else None
            eff_desc = description.strip() if description and description.strip() else None
            eff_name = attendee_name.strip() if attendee_name and attendee_name.strip() else None
            eff_email = attendee_email.strip() if attendee_email and attendee_email.strip() else None

            query = """
                INSERT INTO appointments (
                    lead_id, phone_number, title, event_date, event_time, slot,
                    project_name, external_event_id, description, attendee_name,
                    attendee_email, timezone, status, created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                RETURNING *;
            """
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    query,
                    effective_lead_id,
                    clean_phone,
                    eff_title,
                    eff_date,
                    eff_time,
                    eff_slot,
                    eff_project,
                    external_event_id,
                    eff_desc,
                    eff_name,
                    eff_email,
                    timezone_str,
                    status,
                    now,
                    now,
                )
                if row:
                    appt_dict = dict(row)
                    logger.info(
                        "Created appointment ID %s in PostgreSQL for caller (phone: %s, external_id: %s)",
                        appt_dict.get("id"),
                        mask_phone_number(clean_phone),
                        external_event_id,
                    )
                    return appt_dict
                return None
        except Exception as exc:
            logger.warning("Failed to create appointment in PostgreSQL: %s", type(exc).__name__)
            return None

    async def get_appointments_by_phone(
        self,
        phone_number: Optional[str],
        status: Optional[str] = "confirmed",
    ) -> List[Dict[str, Any]]:
        """Retrieve active appointments for a caller phone number using parameterized SQL."""
        clean_phone = normalize_lead_phone(phone_number)
        if not clean_phone:
            logger.debug("get_appointments_by_phone skipped: invalid or missing phone number.")
            return []

        try:
            pool = await self.pool.get_pool()
            query = """
                SELECT id, lead_id, phone_number, title, event_date, event_time, slot,
                       project_name, external_event_id, description, attendee_name,
                       attendee_email, timezone, status, created_at, updated_at
                FROM appointments
                WHERE phone_number = $1
                  AND ($2::varchar IS NULL OR status = $2)
                ORDER BY created_at DESC;
            """
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, clean_phone, status)
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed to fetch appointments by phone from PostgreSQL: %s", type(exc).__name__)
            return []

    async def get_appointment_by_id(
        self,
        appointment_id: int,
        phone_number: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve an appointment by ID with optional phone ownership filter."""
        if not appointment_id:
            return None

        clean_phone = normalize_lead_phone(phone_number) if phone_number else None
        try:
            pool = await self.pool.get_pool()
            if clean_phone:
                query = """
                    SELECT id, lead_id, phone_number, title, event_date, event_time, slot,
                           project_name, external_event_id, description, attendee_name,
                           attendee_email, timezone, status, created_at, updated_at
                    FROM appointments
                    WHERE id = $1 AND phone_number = $2
                    LIMIT 1;
                """
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(query, appointment_id, clean_phone)
                    return dict(row) if row else None
            else:
                query = """
                    SELECT id, lead_id, phone_number, title, event_date, event_time, slot,
                           project_name, external_event_id, description, attendee_name,
                           attendee_email, timezone, status, created_at, updated_at
                    FROM appointments
                    WHERE id = $1
                    LIMIT 1;
                """
                async with pool.acquire() as conn:
                    row = await conn.fetchrow(query, appointment_id)
                    return dict(row) if row else None
        except Exception as exc:
            logger.warning("Failed to fetch appointment %s from PostgreSQL: %s", appointment_id, type(exc).__name__)
            return None

    async def delete_appointment(
        self,
        appointment_id: int,
        phone_number: str,
    ) -> Optional[Dict[str, Any]]:
        """Delete an exact appointment constrained by both its ID and verified caller phone.

        Uses DELETE ... RETURNING within a PostgreSQL transaction to ensure atomic
        verification of row removal.
        """
        clean_phone = normalize_lead_phone(phone_number)
        if not clean_phone or not appointment_id:
            logger.warning("delete_appointment rejected: valid appointment ID and caller phone are required.")
            return None

        try:
            pool = await self.pool.get_pool()
            query = """
                DELETE FROM appointments
                WHERE id = $1 AND phone_number = $2
                RETURNING id, lead_id, phone_number, title, event_date, event_time, slot,
                          project_name, external_event_id, status;
            """
            async with pool.acquire() as conn:
                async with conn.transaction():
                    row = await conn.fetchrow(query, appointment_id, clean_phone)
                    if not row:
                        logger.info(
                            "Appointment deletion matched 0 rows for ID %s and phone %s",
                            appointment_id,
                            mask_phone_number(clean_phone),
                        )
                        return None
                    deleted_dict = dict(row)
                    logger.info(
                        "Successfully deleted appointment row ID %s from PostgreSQL for phone %s (external_id: %s)",
                        appointment_id,
                        mask_phone_number(clean_phone),
                        deleted_dict.get("external_event_id"),
                    )
                    return deleted_dict
        except Exception as exc:
            logger.warning("Failed to delete appointment from PostgreSQL: %s", type(exc).__name__)
            return None

    async def record_reconciliation_inconsistency(
        self,
        appointment_id: Optional[int],
        external_event_id: Optional[str],
        phone_number: str,
        action: str,
        details: str,
    ) -> Optional[int]:
        """Record an inconsistency entry when external and database stores diverge."""
        clean_phone = normalize_lead_phone(phone_number) or "unknown"
        try:
            pool = await self.pool.get_pool()
            now = datetime.now(timezone.utc)
            query = """
                INSERT INTO appointment_reconciliations (
                    appointment_id, external_event_id, phone_number, action, details, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id;
            """
            async with pool.acquire() as conn:
                rec_id = await conn.fetchval(
                    query, appointment_id, external_event_id, clean_phone, action, details, now
                )
                logger.warning(
                    "Recorded appointment reconciliation inconsistency %s for phone %s (action: %s)",
                    rec_id,
                    mask_phone_number(clean_phone),
                    action,
                )
                return rec_id
        except Exception as exc:
            logger.error("Failed to record reconciliation entry to PostgreSQL: %s", type(exc).__name__)
            return None
