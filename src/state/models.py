"""Deterministic conversation state models and transition rules.

Maintains complete conversational context, intent history, slot values,
domain routing, and lifecycle states for the Unified AI Voice Agent.
"""

from enum import Enum
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.core.types import (
    CallMetadata,
    CallStatus,
    ConversationStage,
    DialogueTurn,
    DomainType,
    ToolExecutionResult,
    TurnRole,
)


class CallerProfile(BaseModel):
    """Profile of the caller, supporting partial inbound and enriched outbound data."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(default=None, description="Caller's known or identified full name")
    phone: Optional[str] = Field(default=None, description="Caller phone number (E.164 format)")
    email: Optional[str] = Field(default=None, description="Caller email address")
    company: Optional[str] = Field(default=None, description="Caller company or academic institution")
    caller_type: Optional[str] = Field(
        default=None,
        description="Caller classification: student, job_seeker, corporate_client, or general",
    )
    known_purpose: Optional[str] = Field(
        default=None,
        description="Pre-identified inquiry objective or outreach context",
    )
    campaign: Optional[str] = Field(default=None, description="Campaign code if outbound outreach")
    campaign_objective: Optional[str] = Field(
        default=None,
        description="Goal to accomplish during outbound outreach",
    )

    def is_known(self) -> bool:
        """Return True if any identifying attribute is populated."""
        return bool(self.name or self.email or self.company)

    def has_contact_info(self) -> bool:
        """Return True if phone or email is available."""
        return bool(self.phone or self.email)


# Explicit phrases that signal the caller wishes to conclude the session
EXPLICIT_TERMINATION_PHRASES = [
    "goodbye",
    "bye",
    "bye bye",
    "that's all",
    "thats all",
    "nothing else",
    "no more questions",
    "i am done",
    "i'm done",
    "end the call",
    "hang up",
    "have a good day",
    "talk to you later",
]


class ConversationState(BaseModel):
    """Central deterministic state of an active or completed conversation."""

    model_config = ConfigDict(extra="forbid")

    # Session & Domain Identification
    metadata: CallMetadata = Field(..., description="Underlying call session metadata")
    current_domain: DomainType = Field(..., description="Currently active business domain")
    primary_domain: DomainType = Field(..., description="Initial domain assigned to call")

    # Intent Stack & Preemption
    current_intent: Optional[str] = Field(default=None, description="Currently active user intent")
    current_sub_intent: Optional[str] = Field(default=None, description="Specialized sub-intent if applicable")
    intent_history: List[str] = Field(
        default_factory=list,
        description="Chronological record of previous intents for contextual back-referencing",
    )

    # Conversational Lifecycle
    stage: ConversationStage = Field(
        default=ConversationStage.GREETING,
        description="High-level conversational milestone (independent of current intent)",
    )

    # Caller & Context Memory
    caller: CallerProfile = Field(
        default_factory=CallerProfile,
        description="Caller profile information (known or incrementally extracted)",
    )
    extracted_slots: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured key-value entities collected during conversation",
    )
    current_interest: Optional[str] = Field(
        default=None,
        description="Primary topic, course, or product of active interest",
    )
    history: List[DialogueTurn] = Field(
        default_factory=list,
        description="Transcript turns exchanged in the call",
    )

    # Asynchronous Context & Action Hooks
    pending_question: Optional[str] = Field(
        default=None,
        description="Question asked by agent awaiting caller response",
    )
    pending_action: Optional[str] = Field(
        default=None,
        description="External action queued or in progress",
    )
    last_action: Optional[str] = Field(
        default=None,
        description="Most recently dispatched external tool action",
    )
    last_tool_result: Optional[ToolExecutionResult] = Field(
        default=None,
        description="Result of the most recently executed tool action",
    )

    # Business vs Conversational Status (Strictly Separated)
    business_status: str = Field(
        default="new",
        description="Domain lifecycle status (e.g. new, inquiring, interested, enrolled)",
    )
    conversation_active: bool = Field(
        default=True,
        description="True if audio/dialogue channel remains open; NOT terminated by business status",
    )
    termination_requested: bool = Field(
        default=False,
        description="True if caller or agent explicitly triggered call termination",
    )
    termination_reason: Optional[str] = Field(
        default=None,
        description="Reason or utterance that caused session conclusion",
    )

    # -------------------------------------------------------------------------
    # State Transition & Manipulation Methods
    # -------------------------------------------------------------------------

    def set_intent(
        self,
        new_intent: str,
        sub_intent: Optional[str] = None,
        clear_pending_question: bool = True,
    ) -> None:
        """Update active intent with historical preservation and question preemption."""
        cleaned_intent = new_intent.strip()

        # Preserve prior intent in history if different
        if self.current_intent and self.current_intent != cleaned_intent:
            self.intent_history.append(self.current_intent)

        self.current_intent = cleaned_intent
        self.current_sub_intent = sub_intent.strip() if sub_intent else None

        # Preemption rule: explicit new intent clears or defers prior pending question
        if clear_pending_question:
            self.pending_question = None

    def switch_domain(self, new_domain: DomainType) -> None:
        """Dynamically switch current active business domain while retaining all session context."""
        if self.current_domain != new_domain:
            self.current_domain = new_domain

    def set_stage(self, new_stage: ConversationStage) -> None:
        """Advance or update the conversational stage."""
        self.stage = new_stage
        if new_stage == ConversationStage.COMPLETED:
            self.conversation_active = False

    def update_slot(self, key: str, value: Any, sync_caller: bool = True) -> None:
        """Record or update an extracted slot and optionally synchronize caller profile."""
        clean_key = key.strip()
        self.extracted_slots[clean_key] = value

        if not sync_caller or value is None:
            return

        str_val = str(value).strip()
        if clean_key in ("student_name", "contact_name", "name") and not self.caller.name:
            self.caller.name = str_val
        elif clean_key == "email" and not self.caller.email:
            self.caller.email = str_val
        elif clean_key in ("company_name", "company") and not self.caller.company:
            self.caller.company = str_val
        elif clean_key in ("interest", "product_interest", "target_course"):
            self.current_interest = str_val
    def get_slot(self, key: str, default: Any = None) -> Any:
        """Return a previously collected conversation slot."""
        return self.extracted_slots.get(key.strip(), default)

    def set_pending_question(self, question: Optional[str]) -> None:
        """Register a question asked to the user awaiting response."""
        self.pending_question = question.strip() if question else None

    def set_pending_action(self, action: Optional[str]) -> None:
        """Register an external action queued for execution."""
        self.pending_action = action.strip() if action else None

    def complete_action(
        self, action_name: str, result: Optional[ToolExecutionResult] = None
    ) -> None:
        """Record completion of an external action without terminating conversation."""
        self.last_action = action_name
        self.last_tool_result = result
        self.pending_action = None
        # Guarantees conversation remains open after action unless explicitly terminated
        self.conversation_active = True

    def update_business_status(self, status: str) -> None:
        """Update domain business status without impacting conversational active state."""
        self.business_status = status.strip()

    def request_termination(self, reason: str = "caller_ended_session") -> None:
        """Explicitly conclude conversational session."""
        self.termination_requested = True
        self.conversation_active = False
        self.termination_reason = reason
        self.stage = ConversationStage.COMPLETED
        self.metadata.status = CallStatus.COMPLETED
        self.metadata.end_time = time.time()

    @staticmethod
    def is_explicit_termination(utterance: str) -> bool:
        """Determine if an utterance constitutes an explicit call termination signal."""
        cleaned = utterance.strip().lower()
        if not cleaned:
            return False

        # Guard: Simple "no", "no thanks", or "not really" must NEVER terminate the session
        # Check against pure negative responses
        negative_only_patterns = [
            r"^no[\.!\?]?$",
            r"^no\s+thanks[\.!\?]?$",
            r"^no\s+thank\s+you[\.!\?]?$",
            r"^not\s+really[\.!\?]?$",
            r"^nope[\.!\?]?$",
            r"^nah[\.!\?]?$",
            r"^no\s+i\s+don'?t[\.!\?]?$",
        ]
        for pattern in negative_only_patterns:
            if re.match(pattern, cleaned):
                return False

        # Check for presence of explicit termination phrases
        for phrase in EXPLICIT_TERMINATION_PHRASES:
            if phrase in cleaned:
                return True

        return False

    def record_turn(
        self,
        role: TurnRole,
        content: str,
        citations: Optional[List[str]] = None,
        tool_calls: Optional[List[str]] = None,
        intent: Optional[str] = None,
    ) -> DialogueTurn:
        """Append a dialogue turn to conversation history."""
        turn = DialogueTurn(
            role=role,
            content=content,
            timestamp=time.time(),
            grounded_citations=citations or [],
            tool_calls_executed=tool_calls or [],
            intent_detected=intent,
        )
        self.history.append(turn)
        return turn

    def clone(self) -> "ConversationState":
        """Return a deep copy of the current state for rollback or inspection."""
        return self.model_copy(deep=True)
