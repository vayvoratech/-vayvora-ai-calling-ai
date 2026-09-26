"""Conversation state manager and session lifecycle factory.

Provides factory methods for Inbound and Outbound conversation initialization
and provides session persistence adhering to the MemoryProvider contract.
"""

import time
from typing import Dict, List, Optional

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


class ConversationStateManager(MemoryProvider):
    """Factory and repository for conversation sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationState] = {}

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    def create_inbound_state(
        self,
        call_id: str,
        caller_phone: str,
        domain: DomainType = DomainType.EDUSAAS,
        caller_name: Optional[str] = None,
        caller_email: Optional[str] = None,
        caller_company: Optional[str] = None,
    ) -> ConversationState:
        """Create a fresh conversation state for an incoming call."""
        metadata = CallMetadata(
            call_id=call_id,
            direction=CallDirection.INBOUND,
            primary_domain=domain,
            caller_phone=caller_phone,
            caller_name=caller_name,
            status=CallStatus.ACTIVE,
            start_time=time.time(),
        )

        caller = CallerProfile(
            name=caller_name,
            phone=caller_phone,
            email=caller_email,
            company=caller_company,
            caller_type="unknown",
        )

        state = ConversationState(
            metadata=metadata,
            current_domain=domain,
            primary_domain=domain,
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
        caller_phone: str,
        domain: DomainType,
        caller_name: str,
        campaign_id: str,
        campaign_objective: str,
        caller_email: Optional[str] = None,
        company: Optional[str] = None,
        known_purpose: Optional[str] = None,
    ) -> ConversationState:
        """Create an enriched conversation state for an outbound outreach call."""
        metadata = CallMetadata(
            call_id=call_id,
            direction=CallDirection.OUTBOUND,
            primary_domain=domain,
            caller_phone=caller_phone,
            caller_name=caller_name,
            campaign_id=campaign_id,
            outbound_objective=campaign_objective,
            status=CallStatus.ACTIVE,
            start_time=time.time(),
        )

        caller = CallerProfile(
            name=caller_name,
            phone=caller_phone,
            email=caller_email,
            company=company,
            campaign=campaign_id,
            campaign_objective=campaign_objective,
            known_purpose=known_purpose,
        )

        state = ConversationState(
            metadata=metadata,
            current_domain=domain,
            primary_domain=domain,
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
