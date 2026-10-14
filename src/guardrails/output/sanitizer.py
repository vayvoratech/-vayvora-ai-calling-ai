"""
Output sanitization layer.

Cleans LLM responses before they are passed to downstream
systems such as TTS.

This layer performs deterministic local processing only.
"""

import re

from src.guardrails.output.config import OutputGuardrailConfig


class OutputSanitizer:
    """Sanitize LLM output for safe downstream processing."""

    def __init__(
        self,
        config: OutputGuardrailConfig | None = None,
    ) -> None:
        self.config = config or OutputGuardrailConfig()

    def sanitize(self, text: str) -> str:
        """
        Sanitize an LLM response.

        The sanitizer:
        - validates the input type
        - removes code blocks
        - removes Markdown formatting
        - normalizes whitespace
        - optionally removes URLs
        - enforces the configured maximum length
        """

        if not isinstance(text, str):
            return ""

        text = text.strip()

        if not text:
            return ""

        if self.config.remove_code_blocks:
            text = self._remove_code_blocks(text)

        if self.config.remove_markdown:
            text = self._remove_markdown(text)

        if self.config.remove_urls:
            text = self._remove_urls(text)

        text = self._normalize_whitespace(text)

        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length].rstrip()

        return text

    @staticmethod
    def _remove_code_blocks(text: str) -> str:
        """Remove fenced Markdown code blocks."""

        return re.sub(
            r"```(?:[\w+-]+)?\s*.*?```",
            "",
            text,
            flags=re.DOTALL,
        )

    @staticmethod
    def _remove_markdown(text: str) -> str:
        """Remove common Markdown formatting."""

        # Images: ![alt](url) -> alt
        text = re.sub(
            r"!\[([^\]]*)\]\([^)]+\)",
            r"\1",
            text,
        )

        # Links: [text](url) -> text
        text = re.sub(
            r"\[([^\]]+)\]\([^)]+\)",
            r"\1",
            text,
        )

        # Headers
        text = re.sub(
            r"^\s{0,3}#{1,6}\s*",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Bold / italic / strikethrough
        text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
        text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
        text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", text)
        text = re.sub(r"~~(.*?)~~", r"\1", text)

        # Inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Markdown bullet markers
        text = re.sub(
            r"^\s*[-*+]\s+",
            "",
            text,
            flags=re.MULTILINE,
        )

        # Numbered-list markers
        text = re.sub(
            r"^\s*\d+[.)]\s+",
            "",
            text,
            flags=re.MULTILINE,
        )

        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove HTTP/HTTPS URLs."""

        return re.sub(
            r"https?://\S+",
            "",
            text,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace for downstream TTS."""

        # Replace tabs and repeated spaces.
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse excessive blank lines.
        text = re.sub(r"\n{2,}", "\n", text)

        return text.strip()