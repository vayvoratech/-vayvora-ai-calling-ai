from src.agents.tools.tool_selector import ToolSelector


def test_select_calendar_events():
    selector = ToolSelector()

    result = selector.select("What's on my calendar?")

    assert result is not None
    assert result.tool_name == "calendar_get_events"
    assert result.confidence > 0.70


def test_select_send_email():
    selector = ToolSelector()

    result = selector.select("Send an email")

    assert result is not None
    assert result.tool_name == "mail_send"


def test_select_whatsapp():
    selector = ToolSelector()

    result = selector.select("Send a WhatsApp message")

    assert result is not None
    assert result.tool_name == "whatsapp_send_message"


def test_no_tool_for_normal_conversation():
    selector = ToolSelector()

    result = selector.select("How are you doing today?")

    assert result is None