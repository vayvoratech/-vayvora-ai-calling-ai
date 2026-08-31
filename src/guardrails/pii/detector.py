"""
Low-latency PII detector.

Uses deterministic regular expressions only.
No LLM calls, network requests, or database access.
"""

import re

from src.guardrails.pii.config import PIIConfig
from src.guardrails.pii.models import PIIDetection


class PIIDetector:
    """Detect common PII patterns in text."""

    EMAIL_PATTERN = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )

    IP_ADDRESS_PATTERN = re.compile(
        r"\b"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
        r"(?:\."
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}"
        r"\b"
    )

    SSN_PATTERN = re.compile(
        r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
    )

    CREDIT_CARD_PATTERN = re.compile(
        r"(?<!\d)"
        r"(?:\d[ -]?){13,19}"
        r"(?!\d)"
    )

    # Phone detection is deliberately conservative.
    PHONE_PATTERN = re.compile(
        r"""
        (?:
            \+\d{1,3}\s\d{10}
            |
            \+\d{1,3}-\d{10}
            |
            \+\d{1,3}\.\d{10}
            |
            \+\d{1,3}\d{10}
            |
            [6-9]\d{9}
            |
            [6-9]\d{2}\s\d{3}\s\d{4}
            |
            [6-9]\d{2}-\d{3}-\d{4}
            |
            [6-9]\d{2}\.\d{3}\.\d{4}
        )
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        config: PIIConfig | None = None,
    ) -> None:
        self.config = config or PIIConfig()

    def detect(self, text: str) -> list[PIIDetection]:
        """Detect configured PII types."""

        if not isinstance(text, str) or not text:
            return []

        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length]

        detections: list[PIIDetection] = []

        # Detect email.
        if self.config.detect_email:
            detections.extend(
                self._find_matches(
                    text,
                    self.EMAIL_PATTERN,
                    "email",
                )
            )

        # Detect IP first.
        if self.config.detect_ip_address:
            detections.extend(
                self._find_matches(
                    text,
                    self.IP_ADDRESS_PATTERN,
                    "ip_address",
                )
            )

        # Detect SSN.
        if self.config.detect_ssn:
            detections.extend(
                self._find_matches(
                    text,
                    self.SSN_PATTERN,
                    "ssn",
                )
            )

        # Detect credit cards.
        if self.config.detect_credit_card:
            detections.extend(
                self._find_matches(
                    text,
                    self.CREDIT_CARD_PATTERN,
                    "credit_card",
                    self._is_valid_card_candidate,
                )
            )

        # Detect phones.
        if self.config.detect_phone:
            detections.extend(
                self._find_phone_matches(text, detections)
            )

        return self._remove_overlaps(detections)

    def _find_phone_matches(
        self,
        text: str,
        existing_detections: list[PIIDetection],
    ) -> list[PIIDetection]:
        """
        Detect phone numbers while protecting already detected
        IP/email/SSN/card regions.
        """

        results: list[PIIDetection] = []

        for match in self.PHONE_PATTERN.finditer(text):
            start = match.start()
            end = match.end()
            value = match.group(0)

            # Do not treat a substring inside another PII value
            # as a phone number.
            overlaps = any(
                start < detection.end
                and end > detection.start
                for detection in existing_detections
            )

            if overlaps:
                continue

            results.append(
                PIIDetection(
                    pii_type="phone",
                    value=value,
                    start=start,
                    end=end,
                    confidence=1.0,
                )
            )

        return results

    @staticmethod
    def _find_matches(
        text: str,
        pattern: re.Pattern,
        pii_type: str,
        validator=None,
    ) -> list[PIIDetection]:
        """Find matches for a PII pattern."""

        detections: list[PIIDetection] = []

        for match in pattern.finditer(text):
            value = match.group(0)

            if validator is not None and not validator(value):
                continue

            detections.append(
                PIIDetection(
                    pii_type=pii_type,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                )
            )

        return detections

    @staticmethod
    def _is_valid_card_candidate(value: str) -> bool:
        """Validate a credit-card candidate using Luhn."""

        digits = re.sub(r"\D", "", value)

        if not 13 <= len(digits) <= 19:
            return False

        total = 0

        for index, digit in enumerate(reversed(digits)):
            number = int(digit)

            if index % 2 == 1:
                number *= 2

                if number > 9:
                    number -= 9

            total += number

        return total % 10 == 0

    @staticmethod
    def _remove_overlaps(
        detections: list[PIIDetection],
    ) -> list[PIIDetection]:
        """Remove overlapping detections."""

        priority = {
            "credit_card": 5,
            "ssn": 4,
            "email": 3,
            "ip_address": 2,
            "phone": 1,
        }

        detections.sort(
            key=lambda detection: (
                -priority.get(detection.pii_type, 0),
                -(detection.end - detection.start),
                detection.start,
            )
        )

        selected: list[PIIDetection] = []

        for detection in detections:
            overlaps = any(
                detection.start < existing.end
                and detection.end > existing.start
                for existing in selected
            )

            if not overlaps:
                selected.append(detection)

        selected.sort(
            key=lambda detection: detection.start
        )

        return selected