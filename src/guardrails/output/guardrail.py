"""
Output guardrail orchestration.

Runs after the LLM and before downstream systems such as TTS.
"""

from src.guardrails.output.config import OutputGuardrailConfig
from src.guardrails.output.detector import OutputDetector
from src.guardrails.output.models import (
    OutputDetection,
    OutputGuardrailResult,
)
from src.guardrails.output.sanitizer import OutputSanitizer
from src.guardrails.pii.config import PIIConfig
from src.guardrails.pii.guardrail import PIIGuardrail


class OutputGuardrail:
    """Validate, sanitize, and protect LLM-generated output."""

    def __init__(
        self,
        config: OutputGuardrailConfig | None = None,
        pii_guardrail: PIIGuardrail | None = None,
        detector: OutputDetector | None = None,
    ) -> None:
        self.config = config or OutputGuardrailConfig()
        self.sanitizer = OutputSanitizer(self.config)
        self.detector = detector or OutputDetector()

        self.pii_guardrail = (
            pii_guardrail
            if pii_guardrail is not None
            else PIIGuardrail(PIIConfig())
        )

    def check(self, text: str) -> OutputGuardrailResult:
        original_text = text if isinstance(text, str) else ""

        detections: list[OutputDetection] = []

        # ---------------------------------------------------------
        # 1. Empty output
        # ---------------------------------------------------------

        if not original_text.strip():
            detections.append(
                OutputDetection(
                    category="empty_output",
                    severity="medium",
                    reason="LLM returned an empty response.",
                )
            )

            return self._blocked_result(
                original_text,
                detections,
                "Empty LLM response.",
            )

        # ---------------------------------------------------------
        # 2. Detect internal information BEFORE sanitization
        # ---------------------------------------------------------

        safety_detections = self.detector.detect(
            text=original_text,
            block_system_prompt_leak=(
                self.config.block_system_prompt_leak
            ),
            block_internal_error_details=(
                self.config.block_internal_error_details
            ),
        )

        detections.extend(safety_detections)

        # Never allow system prompts, credentials, stack traces,
        # internal paths, or connection strings to reach TTS.
        if safety_detections:
            return self._blocked_result(
                original_text,
                detections,
                "Unsafe internal information detected in LLM output.",
            )

        # ---------------------------------------------------------
        # 3. Sanitize presentation format
        # ---------------------------------------------------------

        sanitized_text = self.sanitizer.sanitize(original_text)

        if not sanitized_text:
            detections.append(
                OutputDetection(
                    category="invalid_output",
                    severity="medium",
                    reason="Output became empty after sanitization.",
                )
            )

            return self._blocked_result(
                original_text,
                detections,
                "Invalid output after sanitization.",
            )

        # ---------------------------------------------------------
        # 4. PII detection and redaction
        # ---------------------------------------------------------

        contains_pii = False

        if self.config.enable_pii_redaction:
            pii_result = self.pii_guardrail.check(
                sanitized_text
            )

            contains_pii = pii_result.contains_pii

            if contains_pii:
                for pii_detection in pii_result.detections:
                    detections.append(
                        OutputDetection(
                            category="pii",
                            severity="high",
                            reason=(
                                f"Detected {pii_detection.pii_type} "
                                "in LLM output."
                            ),
                            # Do not expose the full PII value.
                            matched_text="[REDACTED]",
                        )
                    )

                sanitized_text = pii_result.redacted_text

        # Optional strict mode.
        if contains_pii and self.config.block_on_pii:
            return self._blocked_result(
                original_text,
                detections,
                "PII detected in LLM output.",
                contains_pii=True,
            )

        # ---------------------------------------------------------
        # 5. Minimum length
        # ---------------------------------------------------------

        if len(sanitized_text) < self.config.min_text_length:
            detections.append(
                OutputDetection(
                    category="output_length",
                    severity="medium",
                    reason=(
                        "LLM output is shorter than the configured "
                        "minimum length."
                    ),
                )
            )

            return self._blocked_result(
                original_text,
                detections,
                "Output is too short.",
                contains_pii=contains_pii,
            )

        # ---------------------------------------------------------
        # 6. Maximum length
        # ---------------------------------------------------------

        if len(sanitized_text) > self.config.max_text_length:
            sanitized_text = (
                sanitized_text[
                    : self.config.max_text_length
                ].rstrip()
            )

            detections.append(
                OutputDetection(
                    category="output_length",
                    severity="medium",
                    reason=(
                        "LLM output exceeded the configured "
                        "maximum length and was truncated."
                    ),
                )
            )

        # ---------------------------------------------------------
        # 7. Final result
        # ---------------------------------------------------------

        risk_level = self._calculate_risk(detections)

        return OutputGuardrailResult(
            original_text=original_text,
            sanitized_text=sanitized_text,
            is_allowed=True,
            is_blocked=False,
            risk_level=risk_level,
            contains_pii=contains_pii,
            detections=detections,
        )

    def _blocked_result(
        self,
        original_text: str,
        detections: list[OutputDetection],
        reason: str,
        contains_pii: bool = False,
    ) -> OutputGuardrailResult:
        blocked_detections = list(detections)

        blocked_detections.append(
            OutputDetection(
                category="blocked",
                severity="high",
                reason=reason,
            )
        )

        return OutputGuardrailResult(
            original_text=original_text,
            sanitized_text=self.config.fallback_message,
            is_allowed=False,
            is_blocked=True,
            risk_level="high",
            contains_pii=contains_pii,
            detections=blocked_detections,
        )

    @staticmethod
    def _calculate_risk(
        detections: list[OutputDetection],
    ) -> str:
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