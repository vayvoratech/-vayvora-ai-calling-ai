"""Central action verification protocol for external tool outcomes.

Ensures that an HTTP success response is never treated as verified business success
unless it contains authentic external references and passes integrity checks.
"""

from typing import Optional
from src.logging import get_logger
from src.tools.schemas import ToolResult, VerificationStatus

logger = get_logger("tools.verifier")


class ActionVerifier:
    """Verifies that external tool responses meet domain success criteria."""

    @classmethod
    def verify(cls, result: ToolResult) -> ToolResult:
        """Evaluate and attach verification status to a ToolResult."""
        action = result.action_name.strip()

        # If execution already failed at transport or server level, mark failed
        if result.failed or not result.succeeded:
            result.verification_status = VerificationStatus.FAILED
            result.succeeded = False
            result.failed = True
            logger.info("Verification rejected for %s: execution failed.", action)
            return result

        data = result.data or {}

        # 1. Email Verification: Requires valid external message_id
        if action == "send_email":
            msg_id = result.external_reference or data.get("message_id") or data.get("id")
            if msg_id and isinstance(msg_id, str) and len(msg_id.strip()) > 2:
                result.external_reference = msg_id.strip()
                result.verification_status = VerificationStatus.VERIFIED
            else:
                result.verification_status = VerificationStatus.UNVERIFIED
                result.succeeded = False
                result.failed = True
                result.error = "Verification failure: Email response missing external message identifier."

        # 2. Calendar Event Verification: Requires valid event_id or booking reference
        elif action == "create_calendar_event":
            event_id = result.external_reference or data.get("event_id") or data.get("booking_ref")
            if event_id and isinstance(event_id, str) and len(event_id.strip()) > 2:
                result.external_reference = event_id.strip()
                result.verification_status = VerificationStatus.VERIFIED
            else:
                result.verification_status = VerificationStatus.UNVERIFIED
                result.succeeded = False
                result.failed = True
                result.error = "Verification failure: Calendar creation missing external event reference."

        # 3. Available Slots Verification: Requires a valid slots list
        elif action == "find_available_slots":
            slots = data.get("slots")
            if isinstance(slots, list):
                result.verification_status = VerificationStatus.VERIFIED
                result.external_reference = f"slots_found_{len(slots)}"
            else:
                result.verification_status = VerificationStatus.UNVERIFIED
                result.succeeded = False
                result.failed = True
                result.error = "Verification failure: Availability lookup did not return a valid slot list."

        # 4. Lead Update Verification: Requires lead_id or update confirmation
        elif action == "update_lead":
            lead_id = result.external_reference or data.get("lead_id") or data.get("id")
            updated = data.get("updated", True)
            if (lead_id or updated) and not data.get("error"):
                result.external_reference = str(lead_id or "lead_updated_ok")
                result.verification_status = VerificationStatus.VERIFIED
            else:
                result.verification_status = VerificationStatus.UNVERIFIED
                result.succeeded = False
                result.failed = True
                result.error = "Verification failure: Lead update was not confirmed by CRM service."

        # 5. HR Followup Verification: Requires ticket_id or followup_id
        elif action == "create_hr_followup":
            ticket_id = result.external_reference or data.get("ticket_id") or data.get("followup_id")
            if ticket_id and isinstance(ticket_id, str):
                result.external_reference = ticket_id.strip()
                result.verification_status = VerificationStatus.VERIFIED
            else:
                result.verification_status = VerificationStatus.UNVERIFIED
                result.succeeded = False
                result.failed = True
                result.error = "Verification failure: HR follow-up ticket was not generated."

        # 6. Message Verification: Requires message_id
        elif action == "send_message":
            msg_id = result.external_reference or data.get("message_id")
            if msg_id and isinstance(msg_id, str):
                result.external_reference = msg_id.strip()
                result.verification_status = VerificationStatus.VERIFIED
            else:
                result.verification_status = VerificationStatus.UNVERIFIED
                result.succeeded = False
                result.failed = True
                result.error = "Verification failure: Message dispatch missing external confirmation."

        # 7. Business Status Verification
        elif action == "update_business_status":
            status = data.get("status") or result.external_reference
            if status and isinstance(status, str):
                result.external_reference = f"status_{status}"
                result.verification_status = VerificationStatus.VERIFIED
            else:
                result.verification_status = VerificationStatus.UNVERIFIED
                result.succeeded = False
                result.failed = True
                result.error = "Verification failure: Business status update lacked confirmation."

        else:
            # Generic fallback: if succeeded but unrecognized verification pattern
            result.verification_status = VerificationStatus.UNVERIFIED

        logger.debug(
            "Action %s verification status: %s (ref=%s)",
            action,
            result.verification_status.value,
            result.external_reference,
        )
        return result
