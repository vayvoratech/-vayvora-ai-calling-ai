"""
Low-latency input sanitizer.

Cleans speech-to-text output without changing the user's meaning.
No LLM or network calls are used.
"""

import re

from src.guardrails.input.config import InputGuardrailConfig


class InputSanitizer:
    """Clean and normalize user input."""

    def __init__(
        self,
        config: InputGuardrailConfig | None = None,
    ) -> None:
        self.config = config or InputGuardrailConfig()

    def sanitize(self, text: str) -> str:
        """
        Normalize user input.

        The sanitizer intentionally performs only safe transformations.
        It does not attempt to rewrite or interpret the user's request.
        """

        if not isinstance(text, str):
            return ""

        result = text

        if self.config.strip_text:
            result = result.strip()

        if self.config.normalize_whitespace:
            result = re.sub(r"\s+", " ", result)

        # Remove null/control characters while preserving normal
        # punctuation and Unicode characters.
        result = "".join(
            character
            for character in result
            if character in "\n\t"
            or ord(character) >= 32
        )

        if self.config.strip_text:
            result = result.strip()

        # Enforce the configured maximum length.
        if len(result) > self.config.max_text_length:
            result = result[: self.config.max_text_length].rstrip()

        return result