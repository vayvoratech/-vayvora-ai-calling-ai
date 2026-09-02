import re
from datetime import datetime, timedelta
from typing import Any


class ToolArgumentParser:
    """
    Fast deterministic argument parser for simple MCP actions.

    No LLM call is made here.
    """

    def parse(
        self,
        tool_name: str,
        user_input: str,
    ) -> dict[str, Any]:
        text = user_input.strip()

        if tool_name == "calendar_add_event":
            return self._parse_calendar_event(text)

        if tool_name == "mail_send":
            return self._parse_email(text)

        if tool_name == "whatsapp_send_message":
            return self._parse_whatsapp(text)

        return {}

    def _parse_calendar_event(
        self,
        text: str,
    ) -> dict[str, Any]:
        title = self._extract_title(text)

        date_str = self._extract_date(text)

        time_str = self._extract_time(text)

        return {
            "title": title,
            "date_str": date_str,
            "time_str": time_str,
        }

    def _parse_email(
        self,
        text: str,
    ) -> dict[str, Any]:
        email_match = re.search(
            r"[\w.+-]+@[\w-]+\.[\w.-]+",
            text,
        )

        return {
            "to": email_match.group(0) if email_match else "",
            "subject": "",
            "body": text,
        }

    def _parse_whatsapp(
        self,
        text: str,
    ) -> dict[str, Any]:
        phone_match = re.search(
            r"(?:\+91[\s-]?)?[6-9]\d{9}",
            text,
        )

        phone = ""

        if phone_match:
            phone = re.sub(
                r"\D",
                "",
                phone_match.group(0),
            )

        return {
            "phone": phone,
            "message": text,
        }

    @staticmethod
    def _extract_title(text: str) -> str:
        patterns = (
            r"(?:schedule|book|create)\s+(?:a\s+)?(.+?)"
            r"\s+(?:tomorrow|today|on\s+\w+|\d{4}-\d{2}-\d{2})",
            r"(?:schedule|book|create)\s+(?:a\s+)?(.+?)"
            r"\s+at\s+\d",
        )

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:
                return match.group(1).strip()

        return "Scheduled Meeting"

    @staticmethod
    def _extract_date(text: str) -> str:
        today = datetime.now().date()

        if re.search(r"\btomorrow\b", text, re.IGNORECASE):
            return str(today + timedelta(days=1))

        if re.search(r"\btoday\b", text, re.IGNORECASE):
            return str(today)

        date_match = re.search(
            r"\b(\d{4}-\d{2}-\d{2})\b",
            text,
        )

        if date_match:
            return date_match.group(1)

        return str(today)

    @staticmethod
    def _extract_time(text: str) -> str:
        match = re.search(
            r"\b(\d{1,2})(?::(\d{2}))?\s*"
            r"(am|pm)\b",
            text,
            re.IGNORECASE,
        )

        if not match:
            return "10:00"

        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3).lower()

        if meridiem == "pm" and hour != 12:
            hour += 12

        if meridiem == "am" and hour == 12:
            hour = 0

        return f"{hour:02d}:{minute:02d}"