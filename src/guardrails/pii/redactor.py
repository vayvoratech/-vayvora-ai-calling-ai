"""
PII redaction layer.

Takes PII detections and replaces sensitive values with
configured tokens.

No LLM calls, network requests, or database access.
"""

from src.guardrails.pii.config import PIIConfig
from src.guardrails.pii.models import PIIDetection


class PIIRedactor:
    """Redact detected PII from text."""

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
        """
        Replace detected PII with configured tokens.

        Processing is performed from right to left so that
        replacing one value does not invalidate positions of
        detections appearing earlier in the text.
        """

        if not isinstance(text, str) or not text:
            return text

        if not detections:
            return text

        result = text

        # Process from the end of the string toward the beginning.
        sorted_detections = sorted(
            detections,
            key=lambda detection: detection.start,
            reverse=True,
        )

        for detection in sorted_detections:
            token = self._get_token(detection.pii_type)

            if token is None:
                continue

            start = detection.start
            end = detection.end

            # Validate positions before modifying the text.
            if start < 0 or end > len(result) or start >= end:
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
        """Return the configured replacement token."""

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
        }

        return tokens.get(pii_type)