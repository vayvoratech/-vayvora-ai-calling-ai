from datetime import datetime, timedelta

from src.agents.tools.argument_parser import ToolArgumentParser


def test_parse_calendar_event():
    parser = ToolArgumentParser()

    result = parser.parse(
        "calendar_add_event",
        "Schedule a meeting tomorrow at 10 AM",
    )

    expected_date = str(
        datetime.now().date() + timedelta(days=1)
    )

    assert result["title"] == "meeting"
    assert result["date_str"] == expected_date
    assert result["time_str"] == "10:00"


def test_parse_calendar_event_with_pm():
    parser = ToolArgumentParser()

    result = parser.parse(
        "calendar_add_event",
        "Schedule a doctor appointment tomorrow at 3:30 PM",
    )

    assert result["title"] == "doctor appointment"
    assert result["time_str"] == "15:30"


def test_parse_email():
    parser = ToolArgumentParser()

    result = parser.parse(
        "mail_send",
        "Send an email to test@example.com",
    )

    assert result["to"] == "test@example.com"


def test_parse_whatsapp():
    parser = ToolArgumentParser()

    result = parser.parse(
        "whatsapp_send_message",
        "Send WhatsApp message to +919876543210",
    )

    assert result["phone"] == "919876543210"