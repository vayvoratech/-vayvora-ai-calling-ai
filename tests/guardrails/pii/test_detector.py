from src.guardrails.pii.detector import PIIDetector


def test_email_detection():
    detector = PIIDetector()

    result = detector.detect(
        "My email is test@example.com"
    )

    assert len(result) == 1
    assert result[0].pii_type == "email"
    assert result[0].value == "test@example.com"


def test_indian_phone_detection():
    detector = PIIDetector()

    result = detector.detect(
        "Call me at +91 9876543210"
    )

    assert len(result) == 1
    assert result[0].pii_type == "phone"
    assert result[0].value == "+91 9876543210"


def test_ip_address_detection():
    detector = PIIDetector()

    result = detector.detect(
        "Server IP is 192.168.1.10"
    )

    assert len(result) == 1
    assert result[0].pii_type == "ip_address"
    assert result[0].value == "192.168.1.10"


def test_email_phone_and_ip_together():
    detector = PIIDetector()

    text = (
        "My email is test@example.com "
        "and my phone is +91 9876543210. "
        "My IP is 192.168.1.10."
    )

    result = detector.detect(text)

    pii_types = {
        detection.pii_type
        for detection in result
    }

    assert "email" in pii_types
    assert "phone" in pii_types
    assert "ip_address" in pii_types


def test_normal_text_has_no_pii():
    detector = PIIDetector()

    result = detector.detect(
        "Hello, I would like to book an appointment."
    )

    assert result == []