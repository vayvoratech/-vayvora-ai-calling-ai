"""
Configuration for the PII protection layer.

The PII layer is designed for low-latency local detection and
redaction of sensitive information.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PIIConfig:
    """Configuration for PII detection and redaction."""

    # ---------------------------------------------------------
    # Detection switches
    # ---------------------------------------------------------

    detect_email: bool = True
    detect_phone: bool = True
    detect_credit_card: bool = True
    detect_ssn: bool = True
    detect_ip_address: bool = True

    # India-specific PII
    detect_pan: bool = True
    detect_aadhaar: bool = True
    detect_ifsc: bool = True
    detect_upi: bool = True
    detect_bank_account: bool = True
    detect_otp: bool = True

    # ---------------------------------------------------------
    # Redaction switches
    # ---------------------------------------------------------

    redact_email: bool = True
    redact_phone: bool = True
    redact_credit_card: bool = True
    redact_ssn: bool = True
    redact_ip_address: bool = True

    redact_pan: bool = True
    redact_aadhaar: bool = True
    redact_ifsc: bool = True
    redact_upi: bool = True
    redact_bank_account: bool = True
    redact_otp: bool = True

    # ---------------------------------------------------------
    # Replacement tokens
    # ---------------------------------------------------------

    email_token: str = "[EMAIL]"
    phone_token: str = "[PHONE]"
    credit_card_token: str = "[CREDIT_CARD]"
    ssn_token: str = "[SSN]"
    ip_address_token: str = "[IP_ADDRESS]"

    pan_token: str = "[PAN]"
    aadhaar_token: str = "[AADHAAR]"
    ifsc_token: str = "[IFSC]"
    upi_token: str = "[UPI]"
    bank_account_token: str = "[BANK_ACCOUNT]"
    otp_token: str = "[OTP]"

    # ---------------------------------------------------------
    # Safety limits
    # ---------------------------------------------------------

    max_text_length: int = 4000