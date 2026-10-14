"""Validation rules and guards for external actions.

Validates registered action names and required arguments before dispatch.
Domain-specific authorization and trusted consent checks must also be enforced
by the conversation engine; model-generated arguments alone are not proof of consent.
"""

from typing import Set

from src.core.decision import is_valid_email
from src.core.errors import InvalidToolArgumentsError, UnsupportedActionError
from src.core.types import DomainType
from src.tools.message_provider import normalize_phone_number
from src.tools.schemas import ValidatedToolRequest


SUPPORTED_ACTIONS: Set[str] = {
    "send_email",
    "find_available_slots",
    "create_calendar_event",
    "update_lead",
    "create_hr_followup",
    "send_message",
    "update_business_status",
    "mail_read_recent",
}


class ActionValidator:
    """Validate action names and required arguments before execution."""

    @staticmethod
    def validate_action(request: ValidatedToolRequest) -> None:
        """Reject unsupported or insufficiently specified action requests."""
        action = request.action_name.strip()
        if action not in SUPPORTED_ACTIONS:
            raise UnsupportedActionError(
                f"Action '{action}' is not supported. "
                f"Supported actions: {sorted(SUPPORTED_ACTIONS)}"
            )

        args = request.arguments or {}
        domain = getattr(request.domain, "value", request.domain)
        domain = str(domain).strip().lower()

        # Infra Builder uses WhatsApp for requested information sharing.
        # Keep the generic email action registered for other compatible domains,
        # but do not permit it in the Infra Builder workflow.
        if action == "send_email":
            if domain == DomainType.INFRA_BUILDER.value:
                raise UnsupportedActionError(
                    "Email sending is disabled for the Infra Builder workflow; "
                    "use an explicitly authorized supported messaging channel."
                )

            recipient = args.get("recipient") or args.get("email") or request.caller_email
            if not isinstance(recipient, str) or not recipient.strip() or not is_valid_email(recipient):
                raise InvalidToolArgumentsError(
                    "Action 'send_email' requires a valid recipient email address."
                )
            subject = args.get("subject")
            body = args.get("body")
            if subject is not None and (not isinstance(subject, str) or not subject.strip()):
                raise InvalidToolArgumentsError("Email subject must be a non-empty string.")
            if body is not None and (not isinstance(body, str) or not body.strip()):
                raise InvalidToolArgumentsError("Email body must be a non-empty string.")

        elif action == "create_calendar_event":
            slot = args.get("slot") or args.get("start_time") or args.get("time")
            if not isinstance(slot, str) or not slot.strip():
                raise InvalidToolArgumentsError(
                    "Calendar safety violation: a specific appointment time is required."
                )

            # Require an explicit boolean confirmation. This is a secondary guard:
            # the engine must independently verify that the caller confirmed the
            # offered slot and that the slot came from the real availability source.
            if args.get("confirmed") is not True:
                raise InvalidToolArgumentsError(
                    "Calendar safety violation: the caller must explicitly confirm "
                    "the appointment slot before booking."
                )

        elif action == "send_message":
            # This action is used for WhatsApp in the Infra Builder workflow.
            recipient = args.get("recipient") or args.get("phone") or request.caller_phone
            if not isinstance(recipient, str) or not recipient.strip():
                raise InvalidToolArgumentsError(
                    "Action 'send_message' requires a recipient phone number."
                )

            normalized_phone = normalize_phone_number(recipient)
            digits = normalized_phone
            if (
                not digits.isdigit()
                or not 10 <= len(digits) <= 15
                or digits.startswith("0")
                or len(set(digits)) == 1
            ):
                raise InvalidToolArgumentsError(
                    "WhatsApp safety violation: recipient phone number is invalid."
                )

            # Engine-side trusted state must enforce consent. This check prevents
            # missing/false consent flags from passing this validation layer.
            # Never treat an LLM-proposed flag as the sole source of authorization.
            if args.get("consent_confirmed") is not True:
                raise InvalidToolArgumentsError(
                    "WhatsApp safety violation: explicit authorization for this "
                    "message has not been confirmed."
                )

            message = args.get("message") or args.get("text") or args.get("body")
            if not isinstance(message, str) or not message.strip():
                raise InvalidToolArgumentsError(
                    "WhatsApp safety violation: message content must be provided."
                )

            # Do not mutate the request here; dispatch code should use the same
            # phone normalization policy as the messaging provider.
            if not normalized_phone:
                raise InvalidToolArgumentsError(
                    "WhatsApp safety violation: recipient phone number could not be normalized."
                )

        elif action == "find_available_slots":
            # A real availability provider must perform the lookup. This validator
            # checks request shape only and does not imply that any slot is available.
            project = args.get("project_name") or args.get("project") or args.get("property")
            if project is not None and (not isinstance(project, str) or not project.strip()):
                raise InvalidToolArgumentsError(
                    "Availability lookup project/property must be a non-empty string when supplied."
                )

        elif action == "update_lead":
            if not args:
                raise InvalidToolArgumentsError(
                    "Action 'update_lead' requires at least one field to update."
                )
            forbidden = {"domain", "whatsapp_consent", "consent_confirmed"}
            if not any(key not in forbidden for key in args):
                raise InvalidToolArgumentsError(
                    "Action 'update_lead' requires at least one actual lead field to update."
                )

        elif action == "create_hr_followup":
            candidate = args.get("candidate_name") or args.get("name") or request.caller_name
            if not isinstance(candidate, str) or not candidate.strip():
                raise InvalidToolArgumentsError(
                    "Action 'create_hr_followup' requires a candidate/contact name or identification."
                )

        elif action == "update_business_status":
            status = args.get("status")
            if not isinstance(status, str) or not status.strip():
                raise InvalidToolArgumentsError(
                    "Action 'update_business_status' requires a non-empty status string."
                )

        elif action == "mail_read_recent":
            if domain == DomainType.INFRA_BUILDER.value:
                raise UnsupportedActionError(
                    "Email-reading actions are disabled for the Infra Builder workflow."
                )
            limit = args.get("limit", 5)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise InvalidToolArgumentsError(
                    "Action 'mail_read_recent' requires limit to be an integer greater than or equal to 1."
                )
