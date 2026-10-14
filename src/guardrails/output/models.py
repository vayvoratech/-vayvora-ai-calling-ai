"""
Data models for the output guardrail layer.

These models define the contract between the LLM response,
output validation, PII protection, and downstream TTS.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OutputDetection:
    """Represents an issue detected in LLM output."""

    category: str
    severity: str
    reason: str
    matched_text: str | None = None


@dataclass
class OutputGuardrailResult:
    """Final result returned by the output guardrail."""

    original_text: str
    sanitized_text: str

    is_allowed: bool = True
    is_blocked: bool = False

    risk_level: str = "low"
    contains_pii: bool = False

    detections: list[OutputDetection] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        """Convert the result into a JSON-friendly dictionary."""

        return {
            "original_text": self.original_text,
            "sanitized_text": self.sanitized_text,
            "is_allowed": self.is_allowed,
            "is_blocked": self.is_blocked,
            "risk_level": self.risk_level,
            "contains_pii": self.contains_pii,
            "detections": [
                {
                    "category": detection.category,
                    "severity": detection.severity,
                    "reason": detection.reason,
                    "matched_text": detection.matched_text,
                }
                for detection in self.detections
            ],
        }