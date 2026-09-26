"""Registry of available tool schemas and parameter definitions."""

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    """Specification of an MCP tool exposed to the AI agent."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Action name identifier")
    description: str = Field(..., description="Description of the action's purpose")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for arguments")


REGISTERED_TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        name="send_email",
        description="Dispatch an email containing course information, syllabus, or corporate materials.",
        parameters={
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject header"},
                "body": {"type": "string", "description": "Body message or template identifier"},
            },
            "required": ["recipient"],
        },
    ),
    ToolDefinition(
        name="find_available_slots",
        description="Lookup calendar availability for academic consultation or sales discovery meetings.",
        parameters={
            "type": "object",
            "properties": {
                "preferred_date": {"type": "string", "description": "Preferred date or timeframe"},
                "meeting_type": {"type": "string", "description": "Type of consultation"},
            },
        },
    ),
    ToolDefinition(
        name="create_calendar_event",
        description="Book a confirmed meeting slot on the calendar once chosen by the caller.",
        parameters={
            "type": "object",
            "properties": {
                "slot": {"type": "string", "description": "Selected confirmed date/time slot"},
                "attendee_name": {"type": "string", "description": "Attendee full name"},
                "attendee_email": {"type": "string", "description": "Attendee email address"},
                "confirmed": {"type": "boolean", "description": "Explicit confirmation from caller"},
            },
            "required": ["slot"],
        },
    ),
    ToolDefinition(
        name="update_lead",
        description="Update CRM lead details, interest tags, or qualification notes.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Updated lead lifecycle status"},
                "notes": {"type": "string", "description": "Counselor or sales notes"},
            },
        },
    ),
    ToolDefinition(
        name="create_hr_followup",
        description="Queue an HR advisory callback for prospective students or job candidates.",
        parameters={
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string", "description": "Candidate or student name"},
                "phone": {"type": "string", "description": "Contact number"},
                "notes": {"type": "string", "description": "Reason for follow-up"},
            },
        },
    ),
    ToolDefinition(
        name="send_message",
        description="Send an SMS or instant message confirmation.",
        parameters={
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Phone number or user ID"},
                "message": {"type": "string", "description": "Message content"},
            },
            "required": ["recipient"],
        },
    ),
    ToolDefinition(
        name="update_business_status",
        description="Update the domain-specific business lifecycle status.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Status (e.g. interested, qualified, enrolled)"},
            },
            "required": ["status"],
        },
    ),
]


def list_registered_tools() -> List[Dict[str, Any]]:
    """Return tool schemas in standard format for LLM function calling."""
    return [t.model_dump() for t in REGISTERED_TOOLS]
