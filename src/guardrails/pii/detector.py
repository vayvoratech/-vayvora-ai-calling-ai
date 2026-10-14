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

    # ---------------------------------------------------------
    # India-specific PII
    # ---------------------------------------------------------

    # Indian PAN:
    # ABCDE1234F
    PAN_PATTERN = re.compile(
        r"(?<![A-Z0-9])"
        r"[A-Z]{5}\d{4}[A-Z]"
        r"(?![A-Z0-9])",
        re.IGNORECASE,
    )

    # Aadhaar:
    # 1234 5678 9012
    # 1234-5678-9012
    # 123456789012
    AADHAAR_PATTERN = re.compile(
        r"(?<!\d)"
        r"(?:\d{4}[\s-]?\d{4}[\s-]?\d{4})"
        r"(?!\d)"
    )

    # Indian IFSC:
    # SBIN0001234
    IFSC_PATTERN = re.compile(
        r"(?<![A-Z0-9])"
        r"[A-Z]{4}0[A-Z0-9]{6}"
        r"(?![A-Z0-9])",
        re.IGNORECASE,
    )

    # UPI IDs.
    #
    # Deliberately restrict the provider/suffix list so that
    # normal email addresses such as test@example.com are NOT
    # incorrectly classified as UPI.
    UPI_PATTERN = re.compile(
        r"(?<![\w.+-])"
        r"[A-Z0-9][A-Z0-9._+-]{1,255}"
        r"@(?:"
        r"upi"
        r"|ybl"
        r"|ibl"
        r"|axl"
        r"|oksbi"
        r"|okaxis"
        r"|okhdfcbank"
        r"|okicici"
        r"|paytm"
        r"|phonepe"
        r"|payu"
        r"|airtel"
        r"|icici"
        r"|sbi"
        r"|hdfcbank"
        r"|axisbank"
        r"|kotak"
        r"|barodampay"
        r"|fbl"
        r"|rbl"
        r"|yesbank"
        r")"
        r"(?![\w.-])",
        re.IGNORECASE,
    )

    # Bank-account detection requires explicit account context
    # to reduce false positives from ordinary numbers.
    BANK_ACCOUNT_CONTEXT_PATTERN = re.compile(
        r"\b(?:"
        r"account(?:\s+number|\s+no\.?)?"
        r"|a/c"
        r"|bank\s+account"
        r")"
        r"\s*(?:is|:|-)?\s*"
        r"(?P<number>\d{9,18})"
        r"\b",
        re.IGNORECASE,
    )

    # OTP detection requires explicit OTP context.
    OTP_PATTERN = re.compile(
        r"\b(?:otp|one[-\s]?time\s+password)"
        r"\s*(?:is|:|-)?\s*"
        r"(?P<code>\d{4,8})\b",
        re.IGNORECASE,
    )

    # ---------------------------------------------------------
    # Phone
    # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Email
        # ---------------------------------------------------------

        if self.config.detect_email:
            detections.extend(
                self._find_matches(
                    text,
                    self.EMAIL_PATTERN,
                    "email",
                )
            )

        # ---------------------------------------------------------
        # IP address
        # ---------------------------------------------------------

        if self.config.detect_ip_address:
            detections.extend(
                self._find_matches(
                    text,
                    self.IP_ADDRESS_PATTERN,
                    "ip_address",
                )
            )

        # ---------------------------------------------------------
        # SSN
        # ---------------------------------------------------------

        if self.config.detect_ssn:
            detections.extend(
                self._find_matches(
                    text,
                    self.SSN_PATTERN,
                    "ssn",
                )
            )

        # ---------------------------------------------------------
        # Credit card
        # ---------------------------------------------------------

        if self.config.detect_credit_card:
            detections.extend(
                self._find_matches(
                    text,
                    self.CREDIT_CARD_PATTERN,
                    "credit_card",
                    self._is_valid_card_candidate,
                )
            )

        # ---------------------------------------------------------
        # PAN
        # ---------------------------------------------------------

        if self.config.detect_pan:
            detections.extend(
                self._find_matches(
                    text,
                    self.PAN_PATTERN,
                    "pan",
                )
            )

        # ---------------------------------------------------------
        # Aadhaar
        # ---------------------------------------------------------

        if self.config.detect_aadhaar:
            detections.extend(
                self._find_matches(
                    text,
                    self.AADHAAR_PATTERN,
                    "aadhaar",
                    self._is_valid_aadhaar_candidate,
                )
            )

        # ---------------------------------------------------------
        # IFSC
        # ---------------------------------------------------------

        if self.config.detect_ifsc:
            detections.extend(
                self._find_matches(
                    text,
                    self.IFSC_PATTERN,
                    "ifsc",
                )
            )

        # ---------------------------------------------------------
        # UPI
        # ---------------------------------------------------------

        if self.config.detect_upi:
            detections.extend(
                self._find_matches(
                    text,
                    self.UPI_PATTERN,
                    "upi",
                )
            )

        # ---------------------------------------------------------
        # Bank account
        # ---------------------------------------------------------

        if self.config.detect_bank_account:
            detections.extend(
                self._find_contextual_matches(
                    text,
                    self.BANK_ACCOUNT_CONTEXT_PATTERN,
                    "bank_account",
                    group_name="number",
                )
            )

        # ---------------------------------------------------------
        # OTP
        # ---------------------------------------------------------

        if self.config.detect_otp:
            detections.extend(
                self._find_contextual_matches(
                    text,
                    self.OTP_PATTERN,
                    "otp",
                    group_name="code",
                )
            )

        # ---------------------------------------------------------
        # Phone
        # ---------------------------------------------------------

        if self.config.detect_phone:
            detections.extend(
                self._find_phone_matches(
                    text,
                    detections,
                )
            )

        return self._remove_overlaps(detections)

    def _find_phone_matches(
        self,
        text: str,
        existing_detections: list[PIIDetection],
    ) -> list[PIIDetection]:
        """
        Detect phone numbers while protecting already detected
        PII regions.
        """

        results: list[PIIDetection] = []

        for match in self.PHONE_PATTERN.finditer(text):
            start = match.start()
            end = match.end()
            value = match.group(0)

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
    def _find_contextual_matches(
        text: str,
        pattern: re.Pattern,
        pii_type: str,
        group_name: str,
    ) -> list[PIIDetection]:
        """
        Find sensitive values only when they appear in a strong
        contextual pattern such as 'account number' or 'OTP'.
        """

        detections: list[PIIDetection] = []

        for match in pattern.finditer(text):
            value_match = match.span(group_name)

            if value_match == (-1, -1):
                continue

            start, end = value_match
            value = text[start:end]

            detections.append(
                PIIDetection(
                    pii_type=pii_type,
                    value=value,
                    start=start,
                    end=end,
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
    def _is_valid_aadhaar_candidate(value: str) -> bool:
        """
        Basic Aadhaar structural validation.

        This does not verify an Aadhaar against UIDAI.
        """

        digits = re.sub(r"\D", "", value)

        if len(digits) != 12:
            return False

        if len(set(digits)) == 1:
            return False

        if digits.startswith("0") or digits.startswith("1"):
            return False

        return True

    @staticmethod
    def _remove_overlaps(
        detections: list[PIIDetection],
    ) -> list[PIIDetection]:
        """Remove overlapping detections."""

        priority = {
            "credit_card": 10,
            "aadhaar": 9,
            "bank_account": 8,
            "ssn": 7,
            "email": 6,
            "pan": 5,
            "ifsc": 5,
            "upi": 5,
            "otp": 5,
            "ip_address": 3,
            "phone": 2,
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