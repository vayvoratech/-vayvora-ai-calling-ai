from dataclasses import dataclass
from typing import Any

from src.agents.tools.argument_parser import ToolArgumentParser


@dataclass(frozen=True)
class ToolSelection:
    tool_name: str
    tool_input: dict[str, Any]
    confidence: float


class ToolSelector:
    """
    Fast local MCP tool selector.

    Selects the tool and extracts simple arguments
    without making an LLM call.
    """

    RULES = {
        "mail_send": (
            "send email",
            "send an email",
            "email someone",
            "write an email",
        ),
        "mail_read_recent": (
            "read email",
            "read emails",
            "check email",
            "check my email",
            "recent emails",
            "latest emails",
            "show my emails",
        ),
        "calendar_add_event": (
            "schedule",
            "schedule a meeting",
            "schedule an appointment",
            "add to calendar",
            "create calendar event",
            "book a meeting",
        ),
        "calendar_get_events": (
            "calendar",
            "my calendar",
            "calendar events",
            "what is on my calendar",
            "what's on my calendar",
            "show my events",
            "upcoming events",
        ),
        "calendar_get_month_view": (
            "month calendar",
            "calendar for this month",
            "monthly calendar",
        ),
        "whatsapp_send_message": (
            "send whatsapp",
            "send a whatsapp",
            "whatsapp message",
            "message on whatsapp",
        ),
    }

    def __init__(self) -> None:
        self.argument_parser = ToolArgumentParser()

    def select(self, user_input: str) -> ToolSelection | None:
        text = user_input.lower().strip()

        if not text:
            return None

        matches: list[tuple[str, int]] = []

        for tool_name, phrases in self.RULES.items():
            for phrase in phrases:
                if phrase in text:
                    matches.append(
                        (
                            tool_name,
                            len(phrase),
                        )
                    )

        if not matches:
            return None

        tool_name, phrase_length = max(
            matches,
            key=lambda item: item[1],
        )

        tool_input = self.argument_parser.parse(
            tool_name=tool_name,
            user_input=user_input,
        )

        confidence = min(
            0.70 + (phrase_length / 100),
            0.95,
        )

        return ToolSelection(
            tool_name=tool_name,
            tool_input=tool_input,
            confidence=confidence,
        )