"""
PII guardrail orchestration.

Detects and redacts sensitive information using deterministic
local rules.
"""

from src.guardrails.pii.config import PIIConfig
from src.guardrails.pii.detector import PIIDetector
from src.guardrails.pii.models import PIIResult
from src.guardrails.pii.redactor import PIIRedactor


class PIIGuardrail:
    """Detect and redact PII from text."""

    def __init__(
        self,
        config: PIIConfig | None = None,
        detector: PIIDetector | None = None,
        redactor: PIIRedactor | None = None,
    ) -> None:
        self.config = config or PIIConfig()

        self.detector = (
            detector
            if detector is not None
            else PIIDetector(self.config)
        )

        self.redactor = (
            redactor
            if redactor is not None
            else PIIRedactor(self.config)
        )

    def check(self, text: str) -> PIIResult:
        """
        Detect PII and return the redacted text.

        Detection and redaction are completely local.
        """

        original_text = text if isinstance(text, str) else ""

        if not original_text:
            return PIIResult(
                original_text=original_text,
                redacted_text=original_text,
                contains_pii=False,
                detections=[],
            )

        detections = self.detector.detect(
            original_text
        )

        if not detections:
            return PIIResult(
                original_text=original_text,
                redacted_text=original_text,
                contains_pii=False,
                detections=[],
            )

        redacted_text = self.redactor.redact(
            original_text,
            detections,
        )

        return PIIResult(
            original_text=original_text,
            redacted_text=redacted_text,
            contains_pii=True,
            detections=detections,
        )