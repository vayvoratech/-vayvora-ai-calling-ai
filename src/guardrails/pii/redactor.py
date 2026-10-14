"""
PII redaction layer.

Takes PII detections and replaces sensitive values with
configured tokens.

No LLM calls, network requests, or database access.
"""

from src.guardrails.pii.config import PIIConfig
from src.guardrails.pii.models import PIIDetection


class PIIRedactor:
    """Replace detected PII with safe placeholder tokens."""

    def __init__(
        self,
        config: PIIConfig | None = None,
    ) -> None:
        self.config = config or PIIConfig()

    def redact(
        self,
        text: str,
        detections: list[PIIDetection],
    ) -> str:
        if not isinstance(text, str) or not text:
            return text

        if not detections:
            return text

        result = text

        # Replace from right to left so original indexes remain valid.
        sorted_detections = sorted(
            detections,
            key=lambda detection: detection.start,
            reverse=True,
        )

        for detection in sorted_detections:
            token = self._get_token(
                detection.pii_type
            )

            if token is None:
                continue

            start = detection.start
            end = detection.end

            if start < 0:
                continue

            if end > len(result):
                continue

            if start >= end:
                continue

            result = (
                result[:start]
                + token
                + result[end:]
            )

        return result

    def _get_token(
        self,
        pii_type: str,
    ) -> str | None:
        tokens = {
            "email": (
                self.config.email_token
                if self.config.redact_email
                else None
            ),
            "phone": (
                self.config.phone_token
                if self.config.redact_phone
                else None
            ),
            "credit_card": (
                self.config.credit_card_token
                if self.config.redact_credit_card
                else None
            ),
            "ssn": (
                self.config.ssn_token
                if self.config.redact_ssn
                else None
            ),
            "ip_address": (
                self.config.ip_address_token
                if self.config.redact_ip_address
                else None
            ),
            "pan": (
                self.config.pan_token
                if self.config.redact_pan
                else None
            ),
            "aadhaar": (
                self.config.aadhaar_token
                if self.config.redact_aadhaar
                else None
            ),
            "ifsc": (
                self.config.ifsc_token
                if self.config.redact_ifsc
                else None
            ),
            "upi": (
                self.config.upi_token
                if self.config.redact_upi
                else None
            ),
            "bank_account": (
                self.config.bank_account_token
                if self.config.redact_bank_account
                else None
            ),
            "otp": (
                self.config.otp_token
                if self.config.redact_otp
                else None
            ),
        }

        return tokens.get(pii_type)