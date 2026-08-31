from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptResult:
    """
    Standard transcript result shared by all STT providers.
    """

    text: str
    is_final: bool = True
    language: Optional[str] = None
    confidence: Optional[float] = None
    provider: Optional[str] = None

    @property
    def type(self) -> str:
        """
        Message type used by the voice pipeline.

        Final and partial transcripts are both transcript messages.
        """
        return "final" if self.is_final else "partial"

    def to_dict(self) -> dict:
        return {
            "type": "transcript",
            "text": self.text,
            "is_final": self.is_final,
            "language": self.language,
            "confidence": self.confidence,
            "provider": self.provider,
        }


# Common name used by the provider abstraction.
Transcript = TranscriptResult