"""Foundational Pydantic v2 data models and enums.

Defines the core data contracts for the Unified Conversational AI Voice Agent
across EduSaaS and Vayvora domains, inbound and outbound lifecycles.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DomainType(str, Enum):
    """Business domains supported by the unified agent."""

    EDUSAAS = "edusaas"
    VAYVORA = "vayvora"
    GENERAL = "general"


class CallDirection(str, Enum):
    """Direction of telephony or web-based call session."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallStatus(str, Enum):
    """Lifecycle state of an active or past call session."""

    RINGING = "ringing"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    FAILED = "failed"


class TurnRole(str, Enum):
    """Role associated with a conversational turn."""

    CALLER = "caller"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationStage(str, Enum):
    """Controlled stages of the conversational progression."""

    GREETING = "greeting"
    IDENTITY = "identity"
    PURPOSE_DISCOVERY = "purpose_discovery"
    DISCOVERY = "discovery"
    INFORMATION = "information"
    RECOMMENDATION = "recommendation"
    OBJECTION_HANDLING = "objection_handling"
    ACTION_CONFIRMATION = "action_confirmation"
    FOLLOW_UP = "follow_up"
    CLOSING = "closing"
    COMPLETED = "completed"


class DialogueTurn(BaseModel):
    """Represents a single conversational turn in the call transcript."""

    model_config = ConfigDict(extra="forbid")

    role: TurnRole = Field(..., description="Role of the speaker for this turn")
    content: str = Field(..., min_length=1, description="Transcribed or synthesized text")
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp (seconds) when turn was recorded",
    )
    grounded_citations: List[str] = Field(
        default_factory=list,
        description="Identifiers of RAG chunks grounding this turn",
    )
    tool_calls_executed: List[str] = Field(
        default_factory=list,
        description="Names of tools triggered or executed during this turn",
    )
    intent_detected: Optional[str] = Field(
        default=None,
        description="Classified user intent if detected during this turn",
    )


class CallMetadata(BaseModel):
    """Session-level metadata for an inbound or outbound call."""

    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(..., min_length=1, description="Unique identifier for the call session")
    direction: CallDirection = Field(..., description="Inbound or outbound call direction")
    primary_domain: DomainType = Field(..., description="Primary domain assigned to the session")
    caller_phone: str = Field(..., min_length=1, description="Caller or recipient phone number")
    caller_name: Optional[str] = Field(default=None, description="Caller name if known or identified")
    campaign_id: Optional[str] = Field(default=None, description="Campaign ID if outbound call")
    outbound_objective: Optional[str] = Field(
        default=None,
        description="Specific business goal for an outbound outreach call",
    )
    status: CallStatus = Field(default=CallStatus.RINGING, description="Current call lifecycle status")
    start_time: float = Field(
        default_factory=time.time,
        description="Unix timestamp when call connected",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="Unix timestamp when call ended",
    )


class RAGQuery(BaseModel):
    """Input query specification for Grounded RAG retrieval."""

    model_config = ConfigDict(extra="forbid")

    domain: DomainType = Field(..., description="Domain knowledge base to search")
    query_text: str = Field(..., min_length=1, description="Semantic search query text")
    top_k: int = Field(default=3, ge=1, le=20, description="Maximum number of context chunks to return")
    relevance_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score required for grounding",
    )


class RAGChunk(BaseModel):
    """Retrieved knowledge chunk from Redis Stack vector store."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., min_length=1, description="Document or chunk unique identifier")
    domain: DomainType = Field(..., description="Associated domain")
    title: str = Field(..., min_length=1, description="Document title or section heading")
    content: str = Field(..., min_length=1, description="Extracted textual content")
    score: float = Field(ge=0.0, le=1.0, description="Relevance / similarity score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional auxiliary attributes (e.g. source, category, updated_at)",
    )


class ToolCallRequest(BaseModel):
    """Invocation request for an external action via Model Context Protocol (MCP)."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, description="Name of the MCP tool to execute")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters supplied to the tool",
    )
    call_id: str = Field(..., min_length=1, description="Associated call session ID")


class ToolExecutionResult(BaseModel):
    """Execution response and verification status from an MCP tool."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1, description="Name of the tool executed")
    success: bool = Field(..., description="True if action succeeded and was verified")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Payload returned by the external service",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error description if execution or verification failed",
    )
    verification_code: Optional[str] = Field(
        default=None,
        description="Confirmation token or tracking code from external service",
    )
