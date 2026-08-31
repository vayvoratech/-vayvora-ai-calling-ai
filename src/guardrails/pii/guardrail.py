"""
Main PII guardrail.

Pipeline:

Input text
    ↓
PII Detector
    ↓
PII Redactor
    ↓
PIIResult
"""

from src.guardrails.pii.config import PIIConfig
from src.guardrails.pii.detector import PIIDetector
from src.guardrails.pii.models import PIIResult
from src.guardrails.pii.redactor import PIIRedactor


class PIIGuardrail:
    """Low-latency PII detection and redaction interface."""

    def __init__(
        self,
        config: PIIConfig | None = None,
    ) -> None:
        self.config = config or PIIConfig()

        self.detector = PIIDetector(self.config)
        self.redactor = PIIRedactor(self.config)

    def check(self, text: str) -> PIIResult:
        """
        Detect and redact PII from the supplied text.
        """

        if not isinstance(text, str):
            text = ""

        detections = self.detector.detect(text)

        redacted_text = self.redactor.redact(
            text,
            detections,
        )

        return PIIResult(
            original_text=text,
            redacted_text=redacted_text,
            contains_pii=bool(detections),
            detections=detections,
        )

    def detect(self, text: str):
        """Expose detection when only detection is required."""

        return self.detector.detect(text)

    def redact(
        self,
        text: str,
    ) -> str:
        """Detect and return redacted text."""

        detections = self.detector.detect(text)

        return self.redactor.redact(
            text,
            detections,
        )