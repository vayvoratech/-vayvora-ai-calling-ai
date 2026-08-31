"""
Main input guardrail.

Flow:

STT transcript
    ↓
InputGuardrail
    ├── Sanitizer
    └── Detector
    ↓
InputGuardrailResult
"""

from src.guardrails.input.config import InputGuardrailConfig
from src.guardrails.input.detector import InputDetector
from src.guardrails.input.models import (
    InputDetection,
    InputGuardrailResult,
)
from src.guardrails.input.sanitizer import InputSanitizer


class InputGuardrail:
    """Low-latency guardrail for incoming user transcripts."""

    def __init__(
        self,
        config: InputGuardrailConfig | None = None,
    ) -> None:
        self.config = config or InputGuardrailConfig()

        self.sanitizer = InputSanitizer(self.config)
        self.detector = InputDetector(self.config)

    def check(self, text: str) -> InputGuardrailResult:
        """
        Sanitize and inspect user input.

        This method is synchronous and performs no network or LLM calls.
        """

        original_text = text if isinstance(text, str) else ""

        sanitized_text = self.sanitizer.sanitize(original_text)

        detections = self.detector.detect(sanitized_text)

        is_blocked = self._should_block(detections)

        risk_level = self._calculate_risk(detections)

        return InputGuardrailResult(
            original_text=original_text,
            sanitized_text=sanitized_text,
            is_allowed=not is_blocked,
            is_blocked=is_blocked,
            risk_level=risk_level,
            detections=detections,
        )

    def _should_block(
        self,
        detections: list[InputDetection],
    ) -> bool:
        """Determine whether input should be blocked."""

        for detection in detections:
            if detection.category == "prompt_injection":
                if self.config.block_prompt_injection:
                    return True

            if detection.category == "unsafe_request":
                if self.config.block_unsafe_requests:
                    return True

            if detection.category == "input_length":
                return True

            if detection.category == "invalid_input":
                return True

        return False

    @staticmethod
    def _calculate_risk(
        detections: list[InputDetection],
    ) -> str:
        """Calculate overall risk level."""

        if not detections:
            return "low"

        severities = {
            detection.severity
            for detection in detections
        }

        if "high" in severities:
            return "high"

        if "medium" in severities:
            return "medium"

        return "low"