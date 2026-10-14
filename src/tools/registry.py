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
        name="send_message",
        description="Dispatch an authorized WhatsApp message to the caller containing verified property brochures, pricing details, or site visit confirmations.",
        parameters={
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Recipient phone number (normalized with country code)"},
                "message": {"type": "string", "description": "WhatsApp message content or property details"},
                "whatsapp_consent": {"type": "boolean", "description": "Caller explicit authorization for WhatsApp communication"},
            },
            "required": ["recipient", "message"],
        },
    ),
    ToolDefinition(
        name="find_available_slots",
        description="Lookup calendar availability for property site visits or real estate consultations.",
        parameters={
            "type": "object",
            "properties": {
                "preferred_date": {"type": "string", "description": "Preferred date or timeframe for site visit"},
                "property_name": {"type": "string", "description": "Name of the project (e.g. Skyline Towers, Green Valley Villas, Urban Heights)"},
                "meeting_type": {"type": "string", "description": "Type of visit (e.g. site_visit, sales_consultation)"},
            },
        },
    ),
    ToolDefinition(
        name="create_calendar_event",
        description="Book a confirmed property site visit slot on the calendar once verified and explicitly confirmed by the caller.",
        parameters={
            "type": "object",
            "properties": {
                "slot": {"type": "string", "description": "Selected confirmed date/time slot (e.g. '2026-10-15 11:00 AM')"},
                "property_name": {"type": "string", "description": "Name of the project for the site visit"},
                "attendee_name": {"type": "string", "description": "Caller full name"},
                "attendee_phone": {"type": "string", "description": "Caller contact phone number"},
                "confirmed": {"type": "boolean", "description": "Explicit confirmation from caller for the specific slot"},
            },
            "required": ["slot", "confirmed"],
        },
    ),
    ToolDefinition(
        name="update_lead",
        description="Update CRM real estate lead details, preferred project, configuration, budget, or purchase timeline.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Updated lead lifecycle status (e.g. interested, qualified, site_visit_scheduled)"},
                "project_name": {"type": "string", "description": "Preferred real estate project"},
                "configuration": {"type": "string", "description": "Preferred apartment/villa configuration (e.g. 2 BHK, 3 BHK, 4 BHK Villa)"},
                "budget": {"type": "string", "description": "Stated budget or budget range"},
                "notes": {"type": "string", "description": "Customer preferences or qualification notes"},
            },
        },
    ),
    ToolDefinition(
        name="create_hr_followup",
        description="Queue a property relationship manager or sales specialist callback for the caller.",
        parameters={
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string", "description": "Customer contact name"},
                "phone": {"type": "string", "description": "Customer contact phone number"},
                "notes": {"type": "string", "description": "Reason for follow-up or specific questions requiring human assistance"},
            },
            "required": ["candidate_name"],
        },
    ),
    ToolDefinition(
        name="update_business_status",
        description="Update the real estate customer disposition status (e.g. inquiry, site_visit_booked, qualified, follow_up_needed).",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Status string"},
            },
            "required": ["status"],
        },
    ),
]


def list_registered_tools() -> List[Dict[str, Any]]:
    """Return tool schemas in standard format for LLM function calling."""
    return [t.model_dump() for t in REGISTERED_TOOLS]
