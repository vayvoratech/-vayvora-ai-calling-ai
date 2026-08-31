from src.guardrails.pii.guardrail import PIIGuardrail


def test_normal_input_has_no_pii():
    guardrail = PIIGuardrail()

    result = guardrail.check(
        "Hello, I want to book an appointment."
    )

    assert result.contains_pii is False
    assert result.detections == []
    assert result.redacted_text == (
        "Hello, I want to book an appointment."
    )


def test_email_is_detected_and_redacted():
    guardrail = PIIGuardrail()

    result = guardrail.check(
        "My email is test@example.com"
    )

    assert result.contains_pii is True
    assert result.redacted_text == (
        "My email is [EMAIL]"
    )

    assert "test@example.com" not in result.redacted_text


def test_phone_is_detected_and_redacted():
    guardrail = PIIGuardrail()

    result = guardrail.check(
        "Call me at +91 9876543210"
    )

    assert result.contains_pii is True
    assert result.redacted_text == (
        "Call me at [PHONE]"
    )

    assert "+91 9876543210" not in result.redacted_text


def test_multiple_pii_is_redacted():
    guardrail = PIIGuardrail()

    text = (
        "Email test@example.com, "
        "phone +91 9876543210, "
        "IP 192.168.1.10"
    )

    result = guardrail.check(text)

    assert result.contains_pii is True

    assert "test@example.com" not in result.redacted_text
    assert "+91 9876543210" not in result.redacted_text
    assert "192.168.1.10" not in result.redacted_text

    assert "[EMAIL]" in result.redacted_text
    assert "[PHONE]" in result.redacted_text
    assert "[IP_ADDRESS]" in result.redacted_text


def test_result_can_be_converted_to_dict():
    guardrail = PIIGuardrail()

    result = guardrail.check(
        "Contact test@example.com"
    )

    data = result.to_dict()

    assert data["contains_pii"] is True
    assert data["redacted_text"] == (
        "Contact [EMAIL]"
    )
    assert isinstance(data["detections"], list)
    assert len(data["detections"]) == 1