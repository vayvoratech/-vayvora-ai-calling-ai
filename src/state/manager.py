"""Conversation state manager and session lifecycle factory.

Provides factory methods for Inbound and Outbound conversation initialization
and provides session persistence adhering to the MemoryProvider contract.
"""

import time
from typing import Dict, List, Optional
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
from src.state.models import CallerProfile, ConversationState


PROHIBITED_CALLER_PLACEHOLDERS = {
    "default caller",
    "caller@mail",
    "unknown@example.com",
    "john doe",
    "jane doe",
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
    """Factory and repository for conversation sessions with trusted caller profile lookup."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings
        self._sessions: Dict[str, ConversationState] = {}
        self._trusted_profiles: Dict[str, CallerProfile] = {}

    def register_trusted_profile(self, phone: str, profile: CallerProfile) -> None:
        """Register a verified trusted caller profile keyed by phone number."""
        if phone:
            clean_phone = sanitize_caller_value(phone)
            if clean_phone:
                self._trusted_profiles[clean_phone] = profile.model_copy(deep=True)

    def lookup_trusted_profile(self, phone: Optional[str]) -> Optional[CallerProfile]:
        """Lookup an existing verified caller profile by phone number."""
        if not phone:
            return None
        clean_phone = sanitize_caller_value(phone)
        if not clean_phone:
            return None
        profile = self._trusted_profiles.get(clean_phone)
        return profile.model_copy(deep=True) if profile else None

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    def create_inbound_state(
        self,
        call_id: str,
        caller_phone: Optional[str] = None,
        domain: DomainType = DomainType.UNKNOWN,
        caller_name: Optional[str] = None,
        caller_email: Optional[str] = None,
        caller_company: Optional[str] = None,
        domain_locked: bool = False,
    ) -> ConversationState:
        """Create a fresh conversation state for an incoming call.

        Rules:
        - Domain starts as UNKNOWN with domain_locked=False by default.
        - The only caller identity guaranteed at initialization is caller_phone.
        - Never silently invent or populate placeholder names or emails.
        - If a trusted caller profile exists for caller_phone, populate from it.
        - Otherwise, name, email, and company remain None until provided by caller.
        - Phone number is not proof of name or email.
        """
        clean_phone = sanitize_caller_value(caller_phone)
        clean_name = sanitize_caller_value(caller_name)
        clean_email = sanitize_caller_value(caller_email)
        clean_company = sanitize_caller_value(caller_company)

        # Lookup trusted profile by phone if explicit attributes were not supplied
        trusted_profile = self.lookup_trusted_profile(clean_phone) if clean_phone else None
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
            caller_phone=clean_phone,
            caller_name=final_name,
            status=CallStatus.ACTIVE,
            start_time=time.time(),
        )

        caller = CallerProfile(
            name=final_name,
            phone=clean_phone,
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

        self.save(state)
        return state

    def create_outbound_state(
        self,
        call_id: str,
        caller_phone: Optional[str] = None,
        domain: DomainType = DomainType.VAYVORA,
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

    def create_session(
        self,
        session_id: str,
        domain: DomainType = DomainType.UNKNOWN,
        direction: CallDirection = CallDirection.INBOUND,
        caller_profile: Optional[CallerProfile] = None,
        agent_name: Optional[str] = None,
        contact_name: Optional[str] = None,
    ) -> ConversationState:
        """Create or initialize a session respecting inbound/outbound identity rules.
        
        For inbound calls, frontend/API-provided domain must NOT override dynamic routing.
        Initial domain is always UNKNOWN and domain_locked is False.
        For outbound calls, campaign domain context is preserved.
        """
        if direction == CallDirection.OUTBOUND:
            return self.create_outbound_state(
                call_id=session_id,
                caller_phone=caller_profile.phone if caller_profile else None,
                domain=domain if domain != DomainType.UNKNOWN else DomainType.VAYVORA,
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
