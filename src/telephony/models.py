"""Typed telephony event models, structured call summary contracts, and observability telemetry.

Provides validated models for carrier/transport signaling events, end-of-call
persistence summaries, and end-to-end latency diagnostics without exposing secrets.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.core.types import CallDirection, DomainType
from src.state.models import ConversationState


class TelephonyEventType(str, Enum):
    """Telephony signaling and media lifecycle event types."""

    CALL_CONNECTED = "CALL_CONNECTED"
    CALL_DISCONNECTED = "CALL_DISCONNECTED"
    CALL_FAILED = "CALL_FAILED"
    MEDIA_STARTED = "MEDIA_STARTED"
    MEDIA_STOPPED = "MEDIA_STOPPED"
    DTMF_RECEIVED = "DTMF_RECEIVED"


class TelephonyEvent(BaseModel):
    """Discrete signaling or media packet event received from external transport."""

    model_config = ConfigDict(extra="forbid")

    event_type: TelephonyEventType = Field(..., description="Telephony event classification")
    call_id: str = Field(..., min_length=1, description="Unique call identifier")
    timestamp: float = Field(default_factory=time.time, ge=0.0, description="Event Unix timestamp")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload (DTMF digit, disconnect reason, etc.)",
    )


class CallSummary(BaseModel):
    """Structured end-of-call summary persisted at session completion."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Voice session identifier")
    call_id: str = Field(..., description="Telephony call session ID")
    direction: str = Field(..., description="Inbound or outbound call direction")
    domain: str = Field(..., description="Primary or final business domain")
    primary_intent: Optional[str] = Field(default=None, description="Primary recognized intent")
    intent_history: List[str] = Field(default_factory=list, description="All recognized intent transitions")
    caller: Dict[str, Any] = Field(default_factory=dict, description="Identified caller profile metadata")
    business_status: str = Field(default="new", description="CRM/business lead qualification status")
    important_entities: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted domain slots and entities"
    )
    completed_actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Verified external actions dispatched"
    )
    failed_actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Actions that failed or were unverified"
    )
    termination_reason: Optional[str] = Field(
        default=None, description="Reason for call conclusion"
    )
    start_time: float = Field(..., ge=0.0, description="Call start timestamp")
    end_time: float = Field(..., ge=0.0, description="Call end timestamp")
    duration: float = Field(default=0.0, ge=0.0, description="Call duration in seconds")

    @classmethod
    def from_conversation_state(
        cls,
        state: ConversationState,
        session_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        disconnect_reason: Optional[str] = None,
    ) -> "CallSummary":
        """Assemble a verified CallSummary from an active or closed ConversationState."""
        t_start = start_time or state.metadata.start_time
        t_end = end_time or state.metadata.end_time or time.time()
        duration = max(0.0, round(t_end - t_start, 2))

        # Verified completed actions
        completed_actions = []
        failed_actions = []
        if state.last_tool_result:
            success = bool(getattr(state.last_tool_result, "success", False))
            verified = bool(getattr(state.last_tool_result, "verified", success))
            ref = getattr(state.last_tool_result, "verification_code", None) or getattr(
                state.last_tool_result, "external_reference", None
            )
            action_record = {
                "action": state.last_action or "unknown",
                "success": success,
                "verified": verified,
                "reference": ref,
            }
            if success and verified:
                completed_actions.append(action_record)
            else:
                action_record["error"] = getattr(state.last_tool_result, "error_message", None)
                failed_actions.append(action_record)

        caller_dict = {
            "name": state.caller.name,
            "phone": state.caller.phone or state.metadata.caller_phone,
            "email": state.caller.email,
            "company": state.caller.company,
            "caller_type": state.caller.caller_type,
        }
        # Filter out None values
        caller_dict = {k: v for k, v in caller_dict.items() if v is not None}

        # Resolve primary intent
        primary_intent = state.current_intent
        if not primary_intent and state.intent_history:
            primary_intent = state.intent_history[0]

        reason = (
            disconnect_reason
            or state.termination_reason
            or ("completed_goodbye" if not state.conversation_active else "caller_disconnected")
        )

        return cls(
            session_id=session_id or state.metadata.call_id,
            call_id=state.metadata.call_id,
            direction=state.metadata.direction.value,
            domain=state.current_domain.value,
            primary_intent=primary_intent,
            intent_history=list(state.intent_history),
            caller=caller_dict,
            business_status=state.business_status,
            important_entities=dict(state.extracted_slots),
            completed_actions=completed_actions,
            failed_actions=failed_actions,
            termination_reason=reason,
            start_time=round(t_start, 4),
            end_time=round(t_end, 4),
            duration=duration,
        )


class CallObservabilityDiagnostics(BaseModel):
    """End-to-end operational latency and system telemetry for the call session."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    session_id: str
    turn_id: int = 0
    generation_id: int = 0

    # Key Milestones (Unix timestamps)
    connection_time: Optional[float] = None
    first_audio_time: Optional[float] = None
    first_transcript_time: Optional[float] = None
    first_response_time: Optional[float] = None
    first_tts_audio_time: Optional[float] = None
    disconnect_time: Optional[float] = None

    # Telephony Counters & Metrics
    interruptions: int = 0
    stale_frames: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    rag_queries: int = 0
    rag_misses: int = 0
    errors: List[str] = Field(default_factory=list)
