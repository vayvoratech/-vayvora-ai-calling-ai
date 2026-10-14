"""
Deterministic output safety detector.

Detects sensitive internal information that should never be
returned to the caller or passed to downstream systems such as TTS.

No LLM calls, network requests, or database access.
"""

import re

from src.guardrails.output.models import OutputDetection


class OutputDetector:
    """Detect unsafe or internal information in LLM output."""

    SYSTEM_PROMPT_PATTERNS = (
        r"\bsystem prompt\b",
        r"\bsystem message\b",
        r"\bhidden prompt\b",
        r"\binternal instructions?\b",
        r"\bdeveloper instructions?\b",
        r"\bdeveloper message\b",
        r"\bprompt instructions?\b",
        r"\bmy instructions are\b",
        r"\bhere(?:'s| is) my prompt\b",
        r"\bhere(?:'s| is) the system prompt\b",
    )

    INTERNAL_ERROR_PATTERNS = (
        r"\btraceback \(most recent call last\)",
        r"\bexception:\s*",
        r"\bstack trace\b",
        r"\btraceback\b",
        r"\bapi[_ -]?key\s*[:=]",
        r"\bsecret[_ -]?key\s*[:=]",
        r"\bauthentication[_ -]?token\s*[:=]",
        r"\baccess[_ -]?token\s*[:=]",
        r"\bbearer\s+[a-zA-Z0-9._-]+",
        r"\bconnection[_ -]?string\s*[:=]",
        r"\bredis://",
        r"\bpostgres(?:ql)?://",
        r"\bmysql://",
        r"\bfile:///[\w:/\\.-]+",
        r"\b[A-Za-z]:\\(?:[^\\\r\n]+\\)+",
    )

    def detect(
        self,
        text: str,
        block_system_prompt_leak: bool = True,
        block_internal_error_details: bool = True,
    ) -> list[OutputDetection]:
        """Return deterministic safety detections for output text."""

        if not isinstance(text, str) or not text.strip():
            return []

        detections: list[OutputDetection] = []

        if block_system_prompt_leak:
            detections.extend(self._detect_patterns(
                text=text,
                patterns=self.SYSTEM_PROMPT_PATTERNS,
                category="system_prompt_leak",
                severity="high",
                reason="LLM output appears to expose internal prompt or instruction content.",
            ))

        if block_internal_error_details:
            detections.extend(self._detect_patterns(
                text=text,
                patterns=self.INTERNAL_ERROR_PATTERNS,
                category="internal_error",
                severity="high",
                reason="LLM output appears to expose internal error, credential, connection, or filesystem details.",
            ))

        return detections

    @staticmethod
    def _detect_patterns(
        text: str,
        patterns: tuple[str, ...],
        category: str,
        severity: str,
        reason: str,
    ) -> list[OutputDetection]:
        detections: list[OutputDetection] = []

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                detections.append(
                    OutputDetection(
                        category=category,
                        severity=severity,
                        reason=reason,
                        # Do not expose the entire output.
                        matched_text=match.group(0)[:100],
                    )
                )

        return detections