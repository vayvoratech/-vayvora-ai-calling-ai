"""Schemas for validated tool requests and multi-stage tool execution results.

Distinguishes requested, started, succeeded, failed, and verification statuses
to ensure actions are strictly verified before conversational acknowledgment.
"""

from enum import Enum
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.core.types import DomainType, ToolCallRequest, ToolExecutionResult


class VerificationStatus(str, Enum):
    """Lifecycle verification state of an external action result."""

    PENDING = "pending"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"
    FAILED = "failed"


class ValidatedToolRequest(BaseModel):
    """Validated invocation request for an external MCP action."""

    model_config = ConfigDict(extra="forbid")

    action_name: str = Field(..., min_length=1, description="Registered action identifier")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Validated argument payload")
    session_id: str = Field(..., min_length=1, description="Associated call session ID")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request tracing ID",
    )
    domain: DomainType = Field(..., description="Business domain under which action is executed")
    caller_name: Optional[str] = Field(default=None, description="Caller name if known")
    caller_email: Optional[str] = Field(default=None, description="Caller email if known")
    caller_phone: Optional[str] = Field(default=None, description="Caller phone if known")
    intent: Optional[str] = Field(default=None, description="Intent motivating the action")
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Deterministic idempotency token preventing duplicate mutations",
    )

    def to_tool_call_request(self) -> ToolCallRequest:
        """Convert to base Phase 1 ToolCallRequest contract."""
        return ToolCallRequest(
            tool_name=self.action_name,
            arguments=self.arguments,
            call_id=self.session_id,
        )


class ToolResult(BaseModel):
    """Rich execution and verification result from an MCP tool invocation."""

    model_config = ConfigDict(extra="forbid")

    action_name: str = Field(..., min_length=1, description="Name of the action executed")
    requested: bool = Field(default=True, description="True if action was formally requested")
    started: bool = Field(default=False, description="True if execution began on external service")
    succeeded: bool = Field(default=False, description="True if external service returned success response")
    failed: bool = Field(default=False, description="True if external service or transport failed")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Independent verification status of the result",
    )
    external_reference: Optional[str] = Field(
        default=None,
        description="External tracking ID (e.g. message_id, event_id, lead_id)",
    )
    error: Optional[str] = Field(default=None, description="Detailed error description if failed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Response payload from external service")

    @property
    def is_verified_success(self) -> bool:
        """True only if action succeeded AND was independently verified."""
        return self.succeeded and self.verification_status == VerificationStatus.VERIFIED

    def to_tool_execution_result(self) -> ToolExecutionResult:
        """Convert to base Phase 1 ToolExecutionResult contract."""
        return ToolExecutionResult(
            tool_name=self.action_name,
            success=self.is_verified_success,
            data=self.data,
            error_message=self.error,
            verification_code=self.external_reference,
        )
