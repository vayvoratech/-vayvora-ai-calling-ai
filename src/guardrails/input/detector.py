"""
Low-latency input threat detector.

This detector uses deterministic rules and regular expressions.
It does not make network calls or LLM calls.
"""

import re

from src.guardrails.input.config import InputGuardrailConfig
from src.guardrails.input.models import InputDetection


class InputDetector:
    """Detect potentially unsafe or malicious user input."""

    # Common prompt-injection patterns.
    PROMPT_INJECTION_PATTERNS = (
        r"\bignore\s+(all\s+)?previous\s+instructions\b",
        r"\bignore\s+(all\s+)?prior\s+instructions\b",
        r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
        r"\bforget\s+(all\s+)?previous\s+instructions\b",
        r"\boverride\s+(the\s+)?system\s+instructions\b",
        r"\breveal\s+(your\s+)?system\s+prompt\b",
        r"\bshow\s+(me\s+)?(your\s+)?system\s+prompt\b",
        r"\bprint\s+(your\s+)?system\s+prompt\b",
        r"\breveal\s+(your\s+)?hidden\s+instructions\b",
        r"\bshow\s+(me\s+)?your\s+hidden\s+instructions\b",
        r"\bdeveloper\s+message\b",
        r"\bsystem\s+message\b",
        r"\bact\s+as\s+(an?\s+)?(admin|developer|system)\b",
        r"\byou\s+are\s+now\s+(an?\s+)?(admin|developer|system)\b",
        r"\bbypass\s+(the\s+)?(rules|restrictions|guardrails)\b",
        r"\bdisable\s+(the\s+)?(safety|guardrails)\b",
    )

    # Basic unsafe-request patterns.
    UNSAFE_REQUEST_PATTERNS = (
        r"\bhow\s+to\s+make\s+(a\s+)?bomb\b",
        r"\bhow\s+to\s+build\s+(a\s+)?bomb\b",
        r"\bhow\s+to\s+make\s+(a\s+)?weapon\b",
        r"\bhow\s+to\s+build\s+(a\s+)?weapon\b",
        r"\bhow\s+to\s+hack\s+(an?\s+)?account\b",
        r"\bsteal\s+(a\s+)?password\b",
        r"\bsteal\s+(someone'?s\s+)?credentials\b",
    )

    def __init__(
        self,
        config: InputGuardrailConfig | None = None,
    ) -> None:
        self.config = config or InputGuardrailConfig()

        self._injection_patterns = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PROMPT_INJECTION_PATTERNS
        )

        self._unsafe_patterns = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.UNSAFE_REQUEST_PATTERNS
        )

    def detect(self, text: str) -> list[InputDetection]:
        """
        Detect threats in input text.

        Returns a list of detections.
        """

        detections: list[InputDetection] = []

        if not isinstance(text, str):
            detections.append(
                InputDetection(
                    category="invalid_input",
                    severity="high",
                    reason="Input must be a string.",
                )
            )
            return detections

        if not text.strip():
            detections.append(
                InputDetection(
                    category="empty_input",
                    severity="low",
                    reason="Input is empty.",
                )
            )
            return detections

        if len(text) > self.config.max_text_length:
            detections.append(
                InputDetection(
                    category="input_length",
                    severity="medium",
                    reason=(
                        f"Input exceeds maximum length of "
                        f"{self.config.max_text_length} characters."
                    ),
                )
            )

        detections.extend(self._detect_repeated_characters(text))
        detections.extend(self._detect_repeated_words(text))

        if self.config.detect_prompt_injection:
            detections.extend(
                self._detect_patterns(
                    text=text,
                    patterns=self._injection_patterns,
                    category="prompt_injection",
                    severity="high",
                    reason="Potential prompt injection detected.",
                )
            )

        if self.config.detect_unsafe_requests:
            detections.extend(
                self._detect_patterns(
                    text=text,
                    patterns=self._unsafe_patterns,
                    category="unsafe_request",
                    severity="high",
                    reason="Potentially unsafe request detected.",
                )
            )

        detections.extend(self._detect_custom_blocked_phrases(text))

        return detections

    def _detect_patterns(
        self,
        text: str,
        patterns: tuple[re.Pattern[str], ...],
        category: str,
        severity: str,
        reason: str,
    ) -> list[InputDetection]:
        """Detect regex-based threats."""

        detections: list[InputDetection] = []

        for pattern in patterns:
            match = pattern.search(text)

            if match:
                detections.append(
                    InputDetection(
                        category=category,
                        severity=severity,
                        reason=reason,
                        matched_text=match.group(0),
                    )
                )

                # One detection is enough for the same category.
                break

        return detections

    def _detect_repeated_characters(
        self,
        text: str,
    ) -> list[InputDetection]:
        """Detect excessively repeated characters."""

        pattern = re.compile(
            rf"(.)\1{{{self.config.max_repeated_characters},}}"
        )

        match = pattern.search(text)

        if not match:
            return []

        return [
            InputDetection(
                category="repeated_characters",
                severity="low",
                reason="Excessive character repetition detected.",
                matched_text=match.group(0),
            )
        ]

    def _detect_repeated_words(
        self,
        text: str,
    ) -> list[InputDetection]:
        """Detect excessively repeated words."""

        words = re.findall(r"\b[\w']+\b", text.lower())

        if not words:
            return []

        counts: dict[str, int] = {}

        for word in words:
            counts[word] = counts.get(word, 0) + 1

        for word, count in counts.items():
            if count > self.config.max_repeated_words:
                return [
                    InputDetection(
                        category="repeated_words",
                        severity="low",
                        reason="Excessive word repetition detected.",
                        matched_text=word,
                    )
                ]

        return []

    def _detect_custom_blocked_phrases(
        self,
        text: str,
    ) -> list[InputDetection]:
        """Detect application-specific blocked phrases."""

        text_lower = text.lower()

        for phrase in self.config.blocked_phrases:
            if phrase.lower() in text_lower:
                return [
                    InputDetection(
                        category="blocked_phrase",
                        severity="medium",
                        reason="Blocked phrase detected.",
                        matched_text=phrase,
                    )
                ]

        return []