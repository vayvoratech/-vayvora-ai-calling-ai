
"""Registry of available tool schemas and parameter definitions."""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ToolDefinition(BaseModel):
    """Specification of an MCP tool exposed to the AI agent."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Action name identifier")
    description: str = Field(..., description="Description of the action's purpose")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for arguments",
    )


def _object_schema(
    properties: Dict[str, Any],
    required: List[str] | None = None,
) -> Dict[str, Any]:
    """Build a strict JSON schema for tool arguments."""
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }

    if required:
        schema["required"] = required

    return schema


REGISTERED_TOOLS: List[ToolDefinition] = [
    ToolDefinition(
        name="send_message",
        description=(
            "Send an authorized WhatsApp message to a customer. "
            "Use verified property information only. Do not claim the message "
            "was sent unless the tool result confirms the action."
        ),
        parameters=_object_schema(
            {
                "recipient": {
                    "type": "string",
                    "description": "Recipient phone number, preferably with country code.",
                },
                "message": {
                    "type": "string",
                    "description": "Message content to send to the customer.",
                },
                "whatsapp_consent": {
                    "type": "boolean",
                    "description": (
                        "Whether the customer explicitly authorized WhatsApp "
                        "communication. This flag alone is not proof of consent."
                    ),
                },
            },
            required=["recipient", "message"],
        ),
    ),
    ToolDefinition(
        name="find_available_slots",
        description=(
            "Find available calendar slots for a property site visit or "
            "real-estate consultation. Report only availability returned "
            "by the tool."
        ),
        parameters=_object_schema(
            {
                "preferred_date": {
                    "type": "string",
                    "description": "Customer's preferred date or timeframe.",
                },
                "property_name": {
                    "type": "string",
                    "description": "Property project name, if known.",
                },
                "meeting_type": {
                    "type": "string",
                    "description": "Visit type, such as site_visit or sales_consultation.",
                },
            },
        ),
    ),
    ToolDefinition(
        name="create_calendar_event",
        description=(
            "Book a property site visit or consultation only after the customer "
            "has explicitly confirmed the specific slot and the slot has been "
            "verified as available."
        ),
        parameters=_object_schema(
            {
                "slot": {
                    "type": "string",
                    "description": "The selected date and time returned by availability lookup.",
                },
                "property_name": {
                    "type": "string",
                    "description": "Property project associated with the appointment.",
                },
                "attendee_name": {
                    "type": "string",
                    "description": "Customer's name, if known.",
                },
                "attendee_phone": {
                    "type": "string",
                    "description": "Customer's contact number, if known.",
                },
                "confirmed": {
                    "type": "boolean",
                    "description": (
                        "Must be true only after the customer explicitly confirms "
                        "this specific appointment slot."
                    ),
                },
            },
            required=["slot", "confirmed"],
        ),
    ),
    ToolDefinition(
        name="cancel_calendar_event",
        description=(
            "Cancel an existing appointment only after the customer explicitly "
            "confirms the cancellation and the target event has been identified."
        ),
        parameters=_object_schema(
            {
                "event_id": {
                    "type": "string",
                    "description": "Unique identifier of the existing calendar event.",
                },
                "confirmed": {
                    "type": "boolean",
                    "description": (
                        "Must be true only after explicit confirmation to cancel "
                        "the identified appointment."
                    ),
                },
            },
            required=["event_id", "confirmed"],
        ),
    ),
    ToolDefinition(
        name="update_lead",
        description=(
            "Update an existing real-estate CRM lead using customer-provided "
            "or otherwise verified information. Include only fields that "
            "actually need to change."
        ),
        parameters=_object_schema(
            {
                "status": {
                    "type": "string",
                    "enum": [
                        "inquiry",
                        "interested",
                        "qualified",
                        "follow_up_needed",
                        "site_visit_scheduled",
                        "site_visit_booked",
                        "converted",
                        "not_interested",
                        "closed",
                    ],
                    "description": "Updated lead lifecycle status.",
                },
                "project_name": {
                    "type": "string",
                    "description": "Customer's preferred property project.",
                },
                "configuration": {
                    "type": "string",
                    "description": "Preferred configuration, such as 2 BHK or 3 BHK.",
                },
                "budget": {
                    "type": "string",
                    "description": "Customer's stated budget or budget range.",
                },
                "notes": {
                    "type": "string",
                    "description": "Relevant customer preferences or qualification notes.",
                },
            },
        ),
    ),
    ToolDefinition(
        name="create_hr_followup",
        description=(
            "Request a callback from a property relationship manager or sales "
            "specialist. Use verified customer contact details where available."
        ),
        parameters=_object_schema(
            {
                "candidate_name": {
                    "type": "string",
                    "description": "Customer's name, if available.",
                },
                "phone": {
                    "type": "string",
                    "description": "Customer's contact number, if available.",
                },
                "notes": {
                    "type": "string",
                    "description": "Reason for the callback and relevant context.",
                },
            },
        ),
    ),
    ToolDefinition(
        name="update_business_status",
        description=(
            "Update the customer disposition status for the current real-estate "
            "interaction. Use only an allowed status."
        ),
        parameters=_object_schema(
            {
                "status": {
                    "type": "string",
                    "enum": [
                        "inquiry",
                        "interested",
                        "qualified",
                        "follow_up_needed",
                        "site_visit_scheduled",
                        "site_visit_booked",
                        "converted",
                        "not_interested",
                        "closed",
                    ],
                    "description": "New customer disposition status.",
                },
            },
            required=["status"],
        ),
    ),
]


def list_registered_tools() -> List[Dict[str, Any]]:
    """Return tool schemas in standard format for LLM function calling."""
    return [tool.model_dump() for tool in REGISTERED_TOOLS]
