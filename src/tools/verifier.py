
"""Central verification for external tool outcomes.

Only treats a result as verified when the response contains action-specific
confirmation. Transport acceptance is not equivalent to business success.
"""

from typing import Any, Dict, Optional

from src.logging import get_logger
from src.tools.schemas import ToolResult, VerificationStatus

logger = get_logger("tools.verifier")


class ActionVerifier:
    """Validate action-specific evidence before reporting tool success."""

    MOCK_STATUS_VALUES = {
        "mock",
        "mock_sent",
        "mock_confirmed",
        "mock_queued",
        "simulated",
        "simulation",
    }

    MESSAGE_STATUSES = {
        "accepted",
        "queued",
        "sent",
        "delivered",
        "read",
    }

    CALENDAR_SUCCESS_STATUSES = {
        "created",
        "confirmed",
        "booked",
        "scheduled",
    }

    CANCELLATION_SUCCESS_STATUSES = {"cancelled", "canceled"}

    @staticmethod
    def _nonempty_string(value: Any) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return None

    @classmethod
    def _reject(cls, result: ToolResult, message: str) -> ToolResult:
        result.verification_status = VerificationStatus.UNVERIFIED
        result.succeeded = False
        result.failed = True
        result.error = message
        return result

    @classmethod
    def _is_simulated(cls, data: Dict[str, Any]) -> bool:
        """Reject explicit mock or simulation responses."""
        if (
            data.get("mock") is True
            or data.get("is_mock") is True
            or data.get("simulation") is True
        ):
            return True

        status = data.get("status")
        return (
            isinstance(status, str)
            and status.strip().lower() in cls.MOCK_STATUS_VALUES
        )

    @classmethod
    def verify(cls, result: ToolResult) -> ToolResult:
        """Verify an action-specific result and fail closed on weak evidence."""
        action = (result.action_name or "").strip()

        if not action:
            return cls._reject(result, "Verification failure: action name is missing.")

        if result.failed or not result.succeeded:
            result.verification_status = VerificationStatus.FAILED
            result.succeeded = False
            result.failed = True
            if not result.error:
                result.error = "Tool execution failed before verification."
            return result

        data = result.data if isinstance(result.data, dict) else {}

        if cls._is_simulated(data):
            return cls._reject(
                result,
                "Verification failure: simulated result only; "
                "no real external action was verified.",
            )

        if action == "send_email":
            reference = (
                cls._nonempty_string(result.external_reference)
                or cls._nonempty_string(data.get("message_id"))
                or cls._nonempty_string(data.get("id"))
            )
            status = data.get("status")

            # Require a provider reference and an explicit accepted/sent status.
            if (
                not reference
                or not isinstance(status, str)
                or status.strip().lower()
                not in {"accepted", "queued", "sent", "delivered"}
            ):
                return cls._reject(
                    result,
                    "Email verification requires a message reference and "
                    "an accepted or sent status.",
                )

            result.external_reference = reference
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "create_calendar_event":
            reference = (
                cls._nonempty_string(result.external_reference)
                or cls._nonempty_string(data.get("event_id"))
                or cls._nonempty_string(data.get("booking_ref"))
                or cls._nonempty_string(data.get("id"))
            )
            status = data.get("status")
            confirmed = (
                data.get("created") is True
                or data.get("confirmed") is True
                or data.get("success") is True
                or (
                    isinstance(status, str)
                    and status.strip().lower() in cls.CALENDAR_SUCCESS_STATUSES
                )
            )

            if not reference or not confirmed:
                return cls._reject(
                    result,
                    "Calendar verification requires an event reference "
                    "and explicit creation confirmation.",
                )

            result.external_reference = reference
            result.verification_status = VerificationStatus.VERIFIED

        elif action in {"cancel_calendar_event", "calendar_cancel_event"}:
            reference = (
                cls._nonempty_string(result.external_reference)
                or cls._nonempty_string(data.get("appointment_id"))
                or cls._nonempty_string(data.get("event_id"))
                or cls._nonempty_string(data.get("external_event_id"))
            )
            status = data.get("status")
            cancelled = (
                data.get("cancelled") is True
                or (
                    isinstance(status, str)
                    and status.strip().lower()
                    in cls.CANCELLATION_SUCCESS_STATUSES
                )
            )

            if not reference or not cancelled or data.get("error"):
                return cls._reject(
                    result,
                    "Cancellation verification requires a reference and "
                    "explicit confirmation that cancellation succeeded.",
                )

            result.external_reference = reference
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "find_available_slots":
            slots = data.get("slots", data.get("available_slots"))

            if not isinstance(slots, list):
                return cls._reject(
                    result,
                    "Availability verification requires a slots list.",
                )

            result.external_reference = f"slots_found_{len(slots)}"
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "update_lead":
            lead_id = (
                cls._nonempty_string(data.get("lead_id"))
                or cls._nonempty_string(data.get("id"))
            )
            updated = data.get("updated") is True or data.get("success") is True

            if not lead_id or not updated or data.get("error"):
                return cls._reject(
                    result,
                    "CRM verification requires a lead ID and explicit "
                    "confirmation of the update.",
                )

            result.external_reference = lead_id
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "create_hr_followup":
            ticket_id = (
                cls._nonempty_string(data.get("ticket_id"))
                or cls._nonempty_string(data.get("followup_id"))
            )
            status = data.get("status")
            confirmed = (
                data.get("created") is True
                or data.get("success") is True
                or (
                    isinstance(status, str)
                    and status.strip().lower() in {"queued", "created"}
                )
            )

            if not ticket_id or not confirmed or data.get("error"):
                return cls._reject(
                    result,
                    "Follow-up verification requires a ticket reference "
                    "and explicit creation confirmation.",
                )

            result.external_reference = ticket_id
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "send_message":
            status = data.get("status")
            normalized_status = (
                status.strip().lower() if isinstance(status, str) else None
            )
            reference = (
                cls._nonempty_string(data.get("message_id"))
                or cls._nonempty_string(data.get("id"))
            )

            if (
                not reference
                or normalized_status not in cls.MESSAGE_STATUSES
                or data.get("error")
            ):
                return cls._reject(
                    result,
                    "Message verification requires a message reference "
                    "and a valid provider status.",
                )

            result.external_reference = reference
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "update_business_status":
            status = cls._nonempty_string(data.get("status"))
            confirmed = (
                data.get("updated") is True
                or data.get("success") is True
                or data.get("status_confirmed") is True
            )

            if not status or not confirmed or data.get("error"):
                return cls._reject(
                    result,
                    "Business status verification requires explicit "
                    "confirmation of a persisted update.",
                )

            result.external_reference = (
                cls._nonempty_string(data.get("external_reference"))
                or f"status_{status}"
            )
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "mail_read_recent":
            messages = data.get("messages")

            if not isinstance(messages, list) or data.get("error"):
                return cls._reject(
                    result,
                    "Inbox verification requires a valid messages list.",
                )

            result.external_reference = f"emails_read_{len(messages)}"
            result.verification_status = VerificationStatus.VERIFIED

        else:
            return cls._reject(
                result,
                f"Verification failure: no verification rule exists for action '{action}'.",
            )

        logger.debug(
            "Action %s verification status: %s (ref=%s)",
            action,
            result.verification_status.value,
            result.external_reference,
        )
        return result
