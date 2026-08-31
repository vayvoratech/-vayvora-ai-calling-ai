from src.guardrails.pii.detector import PIIDetector
from src.guardrails.pii.redactor import PIIRedactor


def test_email_redaction():
    detector = PIIDetector()
    redactor = PIIRedactor()

    text = "My email is test@example.com"

    detections = detector.detect(text)
    result = redactor.redact(text, detections)

    assert result == "My email is [EMAIL]"
    assert "test@example.com" not in result


def test_phone_redaction():
    detector = PIIDetector()
    redactor = PIIRedactor()

    text = "Call me at +91 9876543210"

    detections = detector.detect(text)
    result = redactor.redact(text, detections)

    assert result == "Call me at [PHONE]"
    assert "+91 9876543210" not in result


def test_ip_redaction():
    detector = PIIDetector()
    redactor = PIIRedactor()

    text = "Server IP is 192.168.1.10"

    detections = detector.detect(text)
    result = redactor.redact(text, detections)

    assert result == "Server IP is [IP_ADDRESS]"
    assert "192.168.1.10" not in result


def test_multiple_pii_redaction():
    detector = PIIDetector()
    redactor = PIIRedactor()

    text = (
        "Email test@example.com, "
        "phone +91 9876543210, "
        "IP 192.168.1.10"
    )

    detections = detector.detect(text)
    result = redactor.redact(text, detections)

    assert "test@example.com" not in result
    assert "+91 9876543210" not in result
    assert "192.168.1.10" not in result

    assert "[EMAIL]" in result
    assert "[PHONE]" in result
    assert "[IP_ADDRESS]" in result


def test_text_without_pii_is_unchanged():
    detector = PIIDetector()
    redactor = PIIRedactor()

    text = "Hello, I want to book an appointment."

    detections = detector.detect(text)
    result = redactor.redact(text, detections)

    assert result == text