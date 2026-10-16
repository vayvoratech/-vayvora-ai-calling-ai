"""Conversation state manager and session lifecycle factory.

Provides factory methods for Inbound and Outbound conversation initialization
with returning-caller profile restoration from PostgreSQL and session persistence
adhering to the MemoryProvider contract.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from src.config import Settings
from src.core.interfaces import MemoryProvider
from src.core.types import (
    CallDirection,
    CallMetadata,
    CallStatus,
    ConversationStage,
    DialogueTurn,
    DomainType,
)
from src.logging import get_logger
from src.state.models import CallerProfile, ConversationState

logger = get_logger("state.manager")

PROHIBITED_CALLER_PLACEHOLDERS = {
    "default caller",
    "caller@mail",
    "unknown@example.com",
    "john doe",
    "jane doe",
    "unknown",
    "none",
    "null",
    "n/a",
    "undefined",
}


def sanitize_caller_value(val: Optional[str]) -> Optional[str]:
    """Strip whitespace and reject any placeholder/fabricated caller values."""
    if not val:
        return None
    cleaned = str(val).strip()
    if not cleaned or cleaned.lower() in PROHIBITED_CALLER_PLACEHOLDERS:
        return None
    return cleaned


class ConversationStateManager(MemoryProvider):
    """Factory and repository for conversation sessions with returning caller profile lookup."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        postgres_repo: Any = "default",
    ) -> None:
        self.settings = settings
        if postgres_repo == "default":
            try:
                from src.database.repository import PostgresRepository
                self.postgres_repo = PostgresRepository()
            except Exception:
                self.postgres_repo = None
        else:
            self.postgres_repo = postgres_repo

        self._sessions: Dict[str, ConversationState] = {}
        self._trusted_profiles: Dict[str, CallerProfile] = {}

    def register_trusted_profile(self, phone: str, profile: CallerProfile) -> None:
        """Register a verified trusted caller profile keyed by normalized phone number."""
        if phone:
            from src.database.repository import normalize_lead_phone
            clean_phone = normalize_lead_phone(phone) or sanitize_caller_value(phone)
            if clean_phone:
                self._trusted_profiles[clean_phone] = profile.model_copy(deep=True)

    async def lookup_trusted_profile_async(
        self, phone: Optional[str], timeout: float = 0.150
    ) -> Optional[CallerProfile]:
        """Lookup an existing verified caller profile by normalized phone number from memory or PostgreSQL."""
        if not phone:
            return None

        from src.database.repository import mask_phone_number, normalize_lead_phone

        clean_phone = normalize_lead_phone(phone)
        if not clean_phone:
            return None

        # 1. In-memory profile check
        profile = self._trusted_profiles.get(clean_phone)
        if profile:
            return profile.model_copy(deep=True)

        # 2. Query PostgreSQL with bounded timeout
        if self.postgres_repo is not None:
            try:
                lead = await asyncio.wait_for(
                    self.postgres_repo.get_lead_by_phone(clean_phone),
                    timeout=timeout,
                )
                if lead:
                    caller_name = sanitize_caller_value(lead.get("name"))
                    caller_email = sanitize_caller_value(lead.get("email"))
                    preferred_proj = sanitize_caller_value(lead.get("interested_in"))
                    caller_type = "known" if (caller_name or caller_email) else "unknown"

                    profile = CallerProfile(
                        name=caller_name,
                        phone=clean_phone,
                        email=caller_email,
                        caller_type=caller_type,
                        known_purpose=preferred_proj,
                    )
                    self._trusted_profiles[clean_phone] = profile
                    logger.info(
                        "Restored returning caller profile from PostgreSQL (phone: %s)",
                        mask_phone_number(clean_phone),
                    )
                    return profile.model_copy(deep=True)
            except asyncio.TimeoutError:
                logger.warning(
                    "PostgreSQL caller lookup timed out (>%.0fms) for phone %s; proceeding as anonymous.",
                    timeout * 1000,
                    mask_phone_number(clean_phone),
                )
            except Exception as exc:
                logger.warning(
                    "PostgreSQL caller lookup failed (%s); proceeding as anonymous.",
                    type(exc).__name__,
                )

        return None

    def lookup_trusted_profile(self, phone: Optional[str]) -> Optional[CallerProfile]:
        """Lookup an existing verified caller profile by phone number (memory + sync DB fallback if no active loop)."""
        if not phone:
            return None
        from src.database.repository import normalize_lead_phone

        clean_phone = normalize_lead_phone(phone) or sanitize_caller_value(phone)
        if not clean_phone:
            return None

        profile = self._trusted_profiles.get(clean_phone)
        if profile:
            return profile.model_copy(deep=True)

        if self.postgres_repo is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is None:
                try:
                    return asyncio.run(self.lookup_trusted_profile_async(clean_phone, timeout=0.150))
                except Exception:
                    pass

        return None

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    async def create_inbound_state_async(
        self,
        call_id: str,
        caller_phone: Optional[str] = None,
        domain: DomainType = DomainType.UNKNOWN,
        caller_name: Optional[str] = None,
        caller_email: Optional[str] = None,
        caller_company: Optional[str] = None,
        domain_locked: bool = False,
        lookup_timeout: float = 0.150,
    ) -> ConversationState:
        """Create a fresh conversation state for an incoming call, restoring caller from PostgreSQL if returning."""
        from src.database.repository import normalize_lead_phone

        clean_phone = normalize_lead_phone(caller_phone) or sanitize_caller_value(caller_phone)

        trusted_profile = None
        if clean_phone and (not caller_name or not caller_email):
            trusted_profile = await self.lookup_trusted_profile_async(clean_phone, timeout=lookup_timeout)

        return self._build_inbound_state(
            call_id=call_id,
            caller_phone=clean_phone,
            domain=domain,
            caller_name=caller_name,
            caller_email=caller_email,
            caller_company=caller_company,
            domain_locked=domain_locked,
            trusted_profile=trusted_profile,
        )

    def create_inbound_state(
        self,
        call_id: str,
        caller_phone: Optional[str] = None,
        domain: DomainType = DomainType.UNKNOWN,
        caller_name: Optional[str] = None,
        caller_email: Optional[str] = None,
        caller_company: Optional[str] = None,
        domain_locked: bool = False,
        caller_profile: Optional[CallerProfile] = None,
    ) -> ConversationState:
        """Create a fresh conversation state for an incoming call."""
        from src.database.repository import normalize_lead_phone

        clean_phone = normalize_lead_phone(caller_phone) or sanitize_caller_value(caller_phone)

        trusted_profile = caller_profile or (self.lookup_trusted_profile(clean_phone) if clean_phone else None)

        return self._build_inbound_state(
            call_id=call_id,
            caller_phone=clean_phone,
            domain=domain,
            caller_name=caller_name,
            caller_email=caller_email,
            caller_company=caller_company,
            domain_locked=domain_locked,
            trusted_profile=trusted_profile,
        )

    def _build_inbound_state(
        self,
        call_id: str,
        caller_phone: Optional[str],
        domain: DomainType,
        caller_name: Optional[str],
        caller_email: Optional[str],
        caller_company: Optional[str],
        domain_locked: bool,
        trusted_profile: Optional[CallerProfile],
    ) -> ConversationState:
        clean_name = sanitize_caller_value(caller_name)
        clean_email = sanitize_caller_value(caller_email)
        clean_company = sanitize_caller_value(caller_company)

        if trusted_profile:
            final_name = clean_name or trusted_profile.name
            final_email = clean_email or trusted_profile.email
            final_company = clean_company or trusted_profile.company
            caller_type = trusted_profile.caller_type or "known"
            known_purpose = trusted_profile.known_purpose
        else:
            final_name = clean_name
            final_email = clean_email
            final_company = clean_company
            caller_type = "unknown"
            known_purpose = None

        metadata = CallMetadata(
            call_id=call_id,
            direction=CallDirection.INBOUND,
            primary_domain=domain,
            caller_phone=caller_phone,
            caller_name=final_name,
            status=CallStatus.ACTIVE,
            start_time=time.time(),
        )

        caller = CallerProfile(
            name=final_name,
            phone=caller_phone,
            email=final_email,
            company=final_company,
            caller_type=caller_type,
            known_purpose=known_purpose,
        )

        state = ConversationState(
            metadata=metadata,
            current_domain=domain,
            primary_domain=domain,
            domain_locked=domain_locked,
            stage=ConversationStage.GREETING,
            caller=caller,
            business_status="new",
            conversation_active=True,
        )

        # Restore verified structured profile slots into current session memory
        if final_name:
            state.extracted_slots["caller_name"] = final_name
        if final_email:
            state.extracted_slots["email"] = final_email
        if known_purpose:
            state.extracted_slots["preferred_project"] = known_purpose
            state.current_interest = known_purpose

        if trusted_profile and (final_name or final_email or known_purpose):
            state.extracted_slots["returning_caller"] = True
            # Identity Safety Gate: phone number match alone does NOT authorize sensitive actions
            state.extracted_slots["returning_caller_verified_sensitive"] = False

        self.save(state)
        return state

    def create_outbound_state(
        self,
        call_id: str,
        caller_phone: Optional[str] = None,
        domain: DomainType = DomainType.INFRA_BUILDER,
        caller_name: Optional[str] = None,
        campaign_id: Optional[str] = None,
        campaign_objective: Optional[str] = None,
        caller_email: Optional[str] = None,
        company: Optional[str] = None,
        known_purpose: Optional[str] = None,
        agent_name: Optional[str] = None,
        contact_name: Optional[str] = None,
        contact_email: Optional[str] = None,
    ) -> ConversationState:
        """Create an enriched conversation state for an outbound outreach call."""
        final_contact = contact_name or caller_name
        final_email = contact_email or caller_email
        clean_phone = sanitize_caller_value(caller_phone)
        clean_name = sanitize_caller_value(final_contact)
        clean_email = sanitize_caller_value(final_email)
        clean_company = sanitize_caller_value(company)
        resolved_agent_name = agent_name or (self.settings.agent_name if self.settings else None)

        metadata = CallMetadata(
            call_id=call_id,
            direction=CallDirection.OUTBOUND,
            primary_domain=domain,
            caller_phone=clean_phone,
            caller_name=clean_name,
            campaign_id=campaign_id,
            outbound_objective=campaign_objective,
            status=CallStatus.ACTIVE,
            start_time=time.time(),
        )

        caller = CallerProfile(
            name=clean_name,
            phone=clean_phone,
            email=clean_email,
            company=clean_company,
            campaign=campaign_id,
            campaign_objective=campaign_objective,
            known_purpose=known_purpose,
        )

        state = ConversationState(
            metadata=metadata,
            current_domain=domain,
            primary_domain=domain,
            domain_locked=True,
            agent_name=resolved_agent_name,
            stage=ConversationStage.GREETING,
            caller=caller,
            business_status="outreach_initiated",
            conversation_active=True,
        )

        self.save(state)
        return state

    # -------------------------------------------------------------------------
    # Repository & Session Storage
    # -------------------------------------------------------------------------

    def save(self, state: ConversationState) -> None:
        """Store or update session state in the repository."""
        self._sessions[state.metadata.call_id] = state

    async def create_session_async(
        self,
        session_id: str,
        domain: DomainType = DomainType.UNKNOWN,
        direction: CallDirection = CallDirection.INBOUND,
        caller_profile: Optional[CallerProfile] = None,
        agent_name: Optional[str] = None,
        contact_name: Optional[str] = None,
        lookup_timeout: float = 0.150,
    ) -> ConversationState:
        """Asynchronously create a session with returning caller lookup for inbound calls."""
        if direction == CallDirection.OUTBOUND:
            return self.create_outbound_state(
                call_id=session_id,
                caller_phone=caller_profile.phone if caller_profile else None,
                domain=domain if domain != DomainType.UNKNOWN else DomainType.INFRA_BUILDER,
                caller_name=caller_profile.name if caller_profile else None,
                campaign_id=caller_profile.campaign if caller_profile else None,
                campaign_objective=caller_profile.campaign_objective if caller_profile else None,
                caller_email=caller_profile.email if caller_profile else None,
                company=caller_profile.company if caller_profile else None,
                known_purpose=caller_profile.known_purpose if caller_profile else None,
                agent_name=agent_name,
                contact_name=contact_name or (caller_profile.name if caller_profile else None),
            )
        else:
            return await self.create_inbound_state_async(
                call_id=session_id,
                caller_phone=caller_profile.phone if caller_profile else None,
                domain=DomainType.UNKNOWN,
                caller_name=caller_profile.name if caller_profile else None,
                caller_email=caller_profile.email if caller_profile else None,
                caller_company=caller_profile.company if caller_profile else None,
                lookup_timeout=lookup_timeout,
            )

    def create_session(
        self,
        session_id: str,
        domain: DomainType = DomainType.UNKNOWN,
        direction: CallDirection = CallDirection.INBOUND,
        caller_profile: Optional[CallerProfile] = None,
        agent_name: Optional[str] = None,
        contact_name: Optional[str] = None,
    ) -> ConversationState:
        """Create or initialize a session respecting inbound/outbound identity rules."""
        if direction == CallDirection.OUTBOUND:
            return self.create_outbound_state(
                call_id=session_id,
                caller_phone=caller_profile.phone if caller_profile else None,
                domain=domain if domain != DomainType.UNKNOWN else DomainType.INFRA_BUILDER,
                caller_name=caller_profile.name if caller_profile else None,
                campaign_id=caller_profile.campaign if caller_profile else None,
                campaign_objective=caller_profile.campaign_objective if caller_profile else None,
                caller_email=caller_profile.email if caller_profile else None,
                company=caller_profile.company if caller_profile else None,
                known_purpose=caller_profile.known_purpose if caller_profile else None,
                agent_name=agent_name,
                contact_name=contact_name or (caller_profile.name if caller_profile else None),
            )
        else:
            return self.create_inbound_state(
                call_id=session_id,
                caller_phone=caller_profile.phone if caller_profile else None,
                domain=DomainType.UNKNOWN,
                caller_name=caller_profile.name if caller_profile else None,
                caller_email=caller_profile.email if caller_profile else None,
                caller_company=caller_profile.company if caller_profile else None,
                caller_profile=caller_profile,
            )

    def get(self, call_id: str) -> Optional[ConversationState]:
        """Fetch session state by unique call ID."""
        return self._sessions.get(call_id)

    def get_or_raise(self, call_id: str) -> ConversationState:
        """Fetch session state, raising KeyError if not found."""
        state = self.get(call_id)
        if not state:
            raise KeyError(f"Session '{call_id}' not found.")
        return state

    def exists(self, call_id: str) -> bool:
        """Check if a session exists."""
        return call_id in self._sessions

    def delete(self, call_id: str) -> bool:
        """Remove a session from storage."""
        return self._sessions.pop(call_id, None) is not None

    def clear(self) -> None:
        """Remove all active sessions (primarily for test isolation)."""
        self._sessions.clear()

    # -------------------------------------------------------------------------
    # MemoryProvider Contract Implementation
    # -------------------------------------------------------------------------

    async def save_turn(self, call_id: str, turn: DialogueTurn) -> None:
        """Record a dialogue turn for the given call ID."""
        state = self.get_or_raise(call_id)
        state.history.append(turn)

    async def get_history(self, call_id: str) -> List[DialogueTurn]:
        """Retrieve ordered dialogue turns for the given call ID."""
        state = self.get(call_id)
        return list(state.history) if state else []

    async def get_metadata(self, call_id: str) -> Optional[CallMetadata]:
        """Retrieve call metadata for the given call ID."""
        state = self.get(call_id)
        return state.metadata if state else None

    async def update_metadata(self, metadata: CallMetadata) -> None:
        """Update existing session metadata."""
        state = self.get_or_raise(metadata.call_id)
        state.metadata = metadata
