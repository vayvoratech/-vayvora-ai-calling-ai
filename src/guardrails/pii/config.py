"""
Configuration for the PII protection layer.

The PII layer is designed for low-latency local detection.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PIIConfig:
    """Configuration for PII detection and redaction."""

    # Detection switches
    detect_email: bool = True
    detect_phone: bool = True
    detect_credit_card: bool = True
    detect_ssn: bool = True
    detect_ip_address: bool = True

    # Redaction behaviour
    redact_email: bool = True
    redact_phone: bool = True
    redact_credit_card: bool = True
    redact_ssn: bool = True
    redact_ip_address: bool = True

    # Replacement format
    email_token: str = "[EMAIL]"
    phone_token: str = "[PHONE]"
    credit_card_token: str = "[CREDIT_CARD]"
    ssn_token: str = "[SSN]"
    ip_address_token: str = "[IP_ADDRESS]"

    # Safety
    max_text_length: int = 4000