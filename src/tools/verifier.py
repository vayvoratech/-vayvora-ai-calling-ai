"""Central verification for external tool outcomes.

Only treats a result as verified when the response contains action-specific
confirmation. Gateway acceptance is not equivalent to recipient delivery.
"""
from typing import Any, Dict, Optional

from src.logging import get_logger
from src.tools.schemas import ToolResult, VerificationStatus

logger = get_logger("tools.verifier")


class ActionVerifier:
    """Validate action-specific evidence before reporting tool success."""

    @staticmethod
    def _nonempty_string(value: Any) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @classmethod
    def _reject(cls, result: ToolResult, message: str) -> ToolResult:
        result.verification_status = VerificationStatus.UNVERIFIED
        result.succeeded = False
        result.failed = True
        result.error = message
        return result

    @classmethod
    def verify(cls, result: ToolResult) -> ToolResult:
        """Attach verification status and fail closed on missing evidence."""
        action = (result.action_name or "").strip()
        if not action:
            return cls._reject(result, "Verification failure: action name is missing.")

        # Never promote a transport/server failure to success.
        if result.failed or not result.succeeded:
            result.verification_status = VerificationStatus.FAILED
            result.succeeded = False
            result.failed = True
            if not result.error:
                result.error = "Tool execution failed before verification."
            logger.info("Verification rejected for %s: execution failed.", action)
            return result

        data: Dict[str, Any] = result.data if isinstance(result.data, dict) else {}

        # Test/mock output is useful in tests, but must never masquerade as a
        # real external action or a customer-facing confirmation.
        if data.get("mock") is True or data.get("is_mock") is True:
            return cls._reject(result, "Mock result only: no real external action was verified.")

        if action == "send_email":
            reference = cls._nonempty_string(result.external_reference) or cls._nonempty_string(data.get("message_id")) or cls._nonempty_string(data.get("id"))
            if not reference:
                return cls._reject(result, "Verification failure: email response lacks an external message identifier.")
            result.external_reference = reference
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "create_calendar_event":
            reference = cls._nonempty_string(result.external_reference) or cls._nonempty_string(data.get("event_id")) or cls._nonempty_string(data.get("booking_ref"))
            if not reference:
                return cls._reject(result, "Verification failure: calendar response lacks an event or booking reference.")
            result.external_reference = reference
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "find_available_slots":
            slots = data.get("slots", data.get("available_slots"))
            if not isinstance(slots, list):
                return cls._reject(result, "Verification failure: availability response must contain a slots list.")
            # An empty list is a valid result: it means no slots were returned.
            result.external_reference = f"slots_found_{len(slots)}"
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "update_lead":
            lead_id = cls._nonempty_string(result.external_reference) or cls._nonempty_string(data.get("lead_id")) or cls._nonempty_string(data.get("id"))
            updated = data.get("updated") is True or data.get("success") is True
            if not lead_id and not updated:
                return cls._reject(result, "Verification failure: CRM response did not confirm the lead update.")
            result.external_reference = lead_id or "lead_updated_confirmed"
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "create_hr_followup":
            ticket_id = cls._nonempty_string(result.external_reference) or cls._nonempty_string(data.get("ticket_id")) or cls._nonempty_string(data.get("followup_id"))
            if not ticket_id:
                return cls._reject(result, "Verification failure: follow-up response lacks a ticket reference.")
            result.external_reference = ticket_id
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "send_message":
            status = data.get("status")
            # mock_sent indicates a simulated message, not a real dispatch.
            if status == "mock_sent":
                return cls._reject(result, "Mock message result only: no real WhatsApp message was dispatched.")
            reference = cls._nonempty_string(result.external_reference) or cls._nonempty_string(data.get("message_id")) or cls._nonempty_string(data.get("id"))
            accepted_statuses = {"accepted", "queued", "sent", "delivered", "read"}
            if not reference or not isinstance(status, str) or status.lower() not in accepted_statuses:
                return cls._reject(result, "Verification failure: message response lacks a valid message reference or accepted status.")
            result.external_reference = reference
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "update_business_status":
            status = cls._nonempty_string(data.get("status"))
            confirmed = data.get("updated") is True or data.get("success") is True
            if not status or data.get("error") or not (confirmed or data.get("status_confirmed") is True):
                return cls._reject(result, "Verification failure: business status response did not explicitly confirm the update.")
            result.external_reference = f"status_{status}"
            result.verification_status = VerificationStatus.VERIFIED

        elif action == "mail_read_recent":
            messages = data.get("messages")
            if not isinstance(messages, list) or data.get("error"):
                return cls._reject(result, "Verification failure: inbox read response did not contain a valid messages list.")
            result.external_reference = f"emails_read_{len(messages)}"
            result.verification_status = VerificationStatus.VERIFIED

        else:
            # Unknown actions must not become successful merely because the
            # transport returned HTTP 200 or a generic success flag.
            result.verification_status = VerificationStatus.UNVERIFIED
            result.succeeded = False
            result.failed = True
            result.error = f"Verification failure: no verification rule exists for action '{action}'."

        logger.debug(
            "Action %s verification status: %s (ref=%s)",
            action,
            result.verification_status.value,
            result.external_reference,
        )
        return result
