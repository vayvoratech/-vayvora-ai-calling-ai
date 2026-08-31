"""
Data models for the PII protection layer.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PIIDetection:
    """Represents a detected piece of personally identifiable information."""

    pii_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class PIIResult:
    """Result returned by the PII detection/redaction pipeline."""

    original_text: str
    redacted_text: str

    contains_pii: bool = False
    detections: list[PIIDetection] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        """Convert the result into a JSON-friendly dictionary."""

        return {
            "original_text": self.original_text,
            "redacted_text": self.redacted_text,
            "contains_pii": self.contains_pii,
            "detections": [
                {
                    "pii_type": detection.pii_type,
                    "value": detection.value,
                    "start": detection.start,
                    "end": detection.end,
                    "confidence": detection.confidence,
                }
                for detection in self.detections
            ],
        }