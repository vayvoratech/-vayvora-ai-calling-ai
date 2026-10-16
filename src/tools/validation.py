"""Validation rules and guards for external actions.

Validates supported actions and their arguments before dispatch.
Trusted authorization and consent must be enforced by the conversation
engine; model-generated arguments alone are not proof of consent.
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
    "cancel_calendar_event",
    "calendar_cancel_event",
    "update_lead",
    "create_hr_followup",
    "send_message",
    "update_business_status",
    "mail_read_recent",
}

# Restrict lifecycle updates to known values.
# Add values here only when the CRM explicitly supports them.
ALLOWED_LEAD_STATUSES: Set[str] = {
    "inquiry",
    "interested",
    "qualified",
    "follow_up_needed",
    "site_visit_scheduled",
    "site_visit_booked",
    "converted",
    "not_interested",
    "closed",
}

ALLOWED_BUSINESS_STATUSES: Set[str] = {
    "inquiry",
    "interested",
    "qualified",
    "follow_up_needed",
    "site_visit_scheduled",
    "site_visit_booked",
    "converted",
    "not_interested",
    "closed",
}

LEAD_FIELDS: Set[str] = {
    "status",
    "project_name",
    "configuration",
    "budget",
    "notes",
}


class ActionValidator:
    """Validate action names, arguments, and safety preconditions."""

    @staticmethod
    def _require_nonempty_string(
        value: object,
        field_name: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidToolArgumentsError(
                f"Action requires a non-empty '{field_name}' string."
            )
        return value.strip()

    @staticmethod
    def _get_domain(request: ValidatedToolRequest) -> str:
        domain = getattr(request.domain, "value", request.domain)
        return str(domain).strip().lower()

    @staticmethod
    def _validate_phone(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidToolArgumentsError(
                "Action 'send_message' requires a recipient phone number."
            )

        normalized = normalize_phone_number(value)
        digits = normalized

        if (
            not isinstance(digits, str)
            or not digits.isdigit()
            or not 10 <= len(digits) <= 15
            or digits.startswith("0")
            or len(set(digits)) == 1
        ):
            raise InvalidToolArgumentsError(
                "WhatsApp safety violation: recipient phone number is invalid."
            )

        return normalized

    @staticmethod
    def validate_action(request: ValidatedToolRequest) -> None:
        """Reject unsupported actions and insufficient or unsafe arguments."""
        action = request.action_name.strip()
        if action not in SUPPORTED_ACTIONS:
            raise UnsupportedActionError(
                f"Action '{action}' is not supported. "
                f"Supported actions: {sorted(SUPPORTED_ACTIONS)}"
            )

        args = request.arguments or {}
        domain = ActionValidator._get_domain(request)

        # Email is disabled for the Infra Builder workflow.
        if action == "send_email":
            if domain == DomainType.INFRA_BUILDER.value:
                raise UnsupportedActionError(
                    "Email sending is disabled for the Infra Builder workflow; "
                    "use an explicitly authorized supported messaging channel."
                )

            recipient = (
                args.get("recipient")
                or args.get("email")
                or request.caller_email
            )

            if (
                not isinstance(recipient, str)
                or not recipient.strip()
                or not is_valid_email(recipient)
            ):
                raise InvalidToolArgumentsError(
                    "Action 'send_email' requires a valid recipient email address."
                )

            subject = args.get("subject")
            body = args.get("body")

            if subject is not None and (
                not isinstance(subject, str) or not subject.strip()
            ):
                raise InvalidToolArgumentsError(
                    "Email subject must be a non-empty string."
                )

            if body is not None and (
                not isinstance(body, str) or not body.strip()
            ):
                raise InvalidToolArgumentsError(
                    "Email body must be a non-empty string."
                )

        elif action == "create_calendar_event":
            slot = (
                args.get("slot")
                or args.get("start_time")
                or args.get("time")
            )

            ActionValidator._require_nonempty_string(slot, "slot")

            if args.get("confirmed") is not True:
                raise InvalidToolArgumentsError(
                    "Calendar safety violation: the caller must explicitly "
                    "confirm the specific appointment slot before booking."
                )

            # Caller identity must not be invented by the model.
            # These fields are optional only if the real calendar integration
            # can resolve the attendee from trusted session state.
            for field in ("property_name", "attendee_name", "attendee_phone"):
                value = args.get(field)
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                ):
                    raise InvalidToolArgumentsError(
                        f"Calendar field '{field}' must be a non-empty string "
                        "when supplied."
                    )

        
        elif action in ("cancel_calendar_event", "calendar_cancel_event"):
            appointment_id = (
                args.get("appointment_id")
                or args.get("event_id")
                or args.get("external_event_id")
            )

            if (
                appointment_id is None
                or isinstance(appointment_id, bool)
                or not isinstance(appointment_id, (str, int))
                or (isinstance(appointment_id, str) and not appointment_id.strip())
                or (isinstance(appointment_id, int) and appointment_id <= 0)
            ):
                raise InvalidToolArgumentsError(
                    "Cancellation safety violation: a valid appointment ID "
                    "or external event ID is required."
                )

            if args.get("confirmed") is not True:
                raise InvalidToolArgumentsError(
                    "Cancellation safety violation: the caller must explicitly "
                    "confirm cancellation before the appointment is deleted."
                )

            # The server must independently verify caller identity and
            # appointment ownership before performing the cancellation.


        elif action == "send_message":
            # Prefer explicit recipient, then known caller phone.
            recipient = args.get("recipient") or args.get("phone") or request.caller_phone
            ActionValidator._validate_phone(recipient)

            message = args.get("message") or args.get("text") or args.get("body")
            ActionValidator._require_nonempty_string(message, "message")

            # Accept the canonical field or the legacy schema field, but never
            # trust either value as consent by itself.
            if (
                args.get("consent_confirmed") is not True
                and args.get("whatsapp_consent") is not True
            ):
                raise InvalidToolArgumentsError(
                    "WhatsApp safety violation: explicit authorization "
                    "for this message has not been confirmed."
                )

            # This is only an argument-level guard. The engine must establish
            # consent from trusted conversation state before dispatch.

        elif action == "find_available_slots":
            preferred_date = args.get("preferred_date")
            if preferred_date is not None and (
                not isinstance(preferred_date, str)
                or not preferred_date.strip()
            ):
                raise InvalidToolArgumentsError(
                    "Availability lookup preferred_date must be a non-empty "
                    "string when supplied."
                )

            project = (
                args.get("property_name")
                or args.get("project_name")
                or args.get("project")
                or args.get("property")
            )

            if project is not None and (
                not isinstance(project, str) or not project.strip()
            ):
                raise InvalidToolArgumentsError(
                    "Availability lookup project name must be a non-empty "
                    "string when supplied."
                )

        elif action == "update_lead":
            # Ignore metadata fields when determining whether an actual
            # lead field was provided.
            provided_fields = set(args.keys()) & LEAD_FIELDS

            if not provided_fields:
                raise InvalidToolArgumentsError(
                    "Action 'update_lead' requires at least one supported "
                    "lead field: status, project_name, configuration, budget, "
                    "or notes."
                )

            unsupported_fields = set(args.keys()) - (
                LEAD_FIELDS
                | {
                    "domain",
                    "caller_name",
                    "caller_email",
                    "caller_phone",
                    "email",
                    "phone",
                    "intent",
                    "consent_confirmed",
                    "whatsapp_consent",
                }
            )

            if unsupported_fields:
                raise InvalidToolArgumentsError(
                    "Unsupported lead update fields: "
                    f"{sorted(unsupported_fields)}"
                )

            for field in provided_fields:
                ActionValidator._require_nonempty_string(
                    args[field],
                    field,
                )

            status = args.get("status")
            if status is not None and status.strip().lower() not in ALLOWED_LEAD_STATUSES:
                raise InvalidToolArgumentsError(
                    f"Unsupported lead status '{status}'. "
                    f"Allowed statuses: {sorted(ALLOWED_LEAD_STATUSES)}"
                )

        elif action == "create_hr_followup":
            candidate = (
                args.get("candidate_name")
                or args.get("name")
                or request.caller_name
            )

            ActionValidator._require_nonempty_string(
                candidate,
                "candidate_name",
            )

            notes = args.get("notes")
            if notes is not None and (
                not isinstance(notes, str) or not notes.strip()
            ):
                raise InvalidToolArgumentsError(
                    "Follow-up notes must be a non-empty string when supplied."
                )

            phone = args.get("phone")
            if phone is not None:
                ActionValidator._validate_phone(phone)

        
        elif action == "update_business_status":
            raise UnsupportedActionError(
                "Business status updates are temporarily disabled because "
                "persistent storage is not implemented for this action."
            )

        elif action == "mail_read_recent":
            if domain == DomainType.INFRA_BUILDER.value:
                raise UnsupportedActionError(
                    "Email-reading actions are disabled for the Infra Builder workflow."
                )

            limit = args.get("limit", 5)
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or limit < 1
                or limit > 100
            ):
                raise InvalidToolArgumentsError(
                    "Action 'mail_read_recent' requires limit to be "
                    "an integer between 1 and 100."
                )