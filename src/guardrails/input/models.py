"""
Data models for the input guardrail layer.

These models define the contract between input detection,
sanitization, and the rest of the AI pipeline.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InputDetection:
    """Represents a single issue detected in user input."""

    category: str
    severity: str
    reason: str
    matched_text: str | None = None


@dataclass
class InputGuardrailResult:
    """Final result returned by the input guardrail."""

    original_text: str
    sanitized_text: str

    is_allowed: bool = True
    is_blocked: bool = False

    risk_level: str = "low"

    detections: list[InputDetection] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        """Convert the result to a JSON-friendly dictionary."""

        return {
            "original_text": self.original_text,
            "sanitized_text": self.sanitized_text,
            "is_allowed": self.is_allowed,
            "is_blocked": self.is_blocked,
            "risk_level": self.risk_level,
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