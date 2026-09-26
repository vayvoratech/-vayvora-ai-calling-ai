"""Validation rules and guards for proposed external actions.

Validates action names and argument schemas before dispatching requests to external
MCP servers, enforcing calendar and email safety invariants.
"""

import re
from typing import Any, Dict, Set
from src.core.errors import InvalidToolArgumentsError, UnsupportedActionError
from src.tools.schemas import ValidatedToolRequest

SUPPORTED_ACTIONS: Set[str] = {
    "send_email",
    "find_available_slots",
    "create_calendar_event",
    "update_lead",
    "create_hr_followup",
    "send_message",
    "update_business_status",
}


class ActionValidator:
    """Validates action names and arguments before execution."""

    @staticmethod
    def validate_action(request: ValidatedToolRequest) -> None:
        """Validate request action name and required arguments."""
        action = request.action_name.strip()
        if action not in SUPPORTED_ACTIONS:
            raise UnsupportedActionError(
                f"Action '{action}' is not supported. Supported actions: {sorted(SUPPORTED_ACTIONS)}"
            )

        args = request.arguments

        # 1. Email Safety Validation
        if action == "send_email":
            recipient = args.get("recipient") or args.get("email") or request.caller_email
            if not recipient or not isinstance(recipient, str) or "@" not in recipient:
                raise InvalidToolArgumentsError(
                    "Action 'send_email' requires a valid recipient email address."
                )

        # 2. Calendar Safety Validation
        elif action == "create_calendar_event":
            slot = args.get("slot") or args.get("start_time") or args.get("time")
            if not slot or not isinstance(slot, str) or not slot.strip():
                raise InvalidToolArgumentsError(
                    "Calendar Safety Violation: Cannot create calendar event without a specified time slot."
                )
            # Require explicit confirmation flag to prevent booking unconfirmed slots
            confirmed = args.get("confirmed", True)
            if not confirmed:
                raise InvalidToolArgumentsError(
                    "Calendar Safety Violation: Cannot create calendar event before caller confirms the slot."
                )

        # 3. Message Safety Validation
        elif action == "send_message":
            recipient = args.get("recipient") or args.get("phone") or request.caller_phone
            if not recipient:
                raise InvalidToolArgumentsError(
                    "Action 'send_message' requires a recipient phone number or contact."
                )

        # 4. Lead Update Validation
        elif action == "update_lead":
            if not args:
                raise InvalidToolArgumentsError(
                    "Action 'update_lead' requires at least one field to update."
                )

        # 5. HR Followup Validation
        elif action == "create_hr_followup":
            candidate = args.get("candidate_name") or args.get("name") or request.caller_name
            if not candidate:
                raise InvalidToolArgumentsError(
                    "Action 'create_hr_followup' requires candidate name or identification."
                )

        # 6. Business Status Validation
        elif action == "update_business_status":
            status = args.get("status")
            if not status or not isinstance(status, str):
                raise InvalidToolArgumentsError(
                    "Action 'update_business_status' requires a valid status string."
                )
