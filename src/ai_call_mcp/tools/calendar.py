import calendar
import json
import os
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP


# Store events next to the MCP package instead of depending
# on the current working directory.
EVENTS_FILE = Path(__file__).resolve().parent.parent / "events.json"


def load_events() -> list[dict]:
    """Load events from local JSON storage."""

    if not EVENTS_FILE.exists():
        return []

    try:
        with EVENTS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def save_events(events: list[dict]) -> None:
    """Save events to local JSON storage."""

    with EVENTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(events, file, indent=4)


def register_calendar_tools(mcp: FastMCP) -> None:
    """Register calendar tools with the MCP server."""

    @mcp.tool()
    async def calendar_add_event(
        title: str,
        date_str: str,
        time_str: str,
        description: str = "",
    ) -> str:
        """
        Schedule a new calendar event.

        date_str format: YYYY-MM-DD
        time_str format: HH:MM
        """

        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            datetime.strptime(time_str, "%H:%M")
        except ValueError:
            raise ValueError(
                "Invalid date or time. "
                "Use YYYY-MM-DD and HH:MM."
            )

        events = load_events()

        new_event = {
            "id": len(events) + 1,
            "title": title,
            "date": date_str,
            "time": time_str,
            "description": description,
            "created_at": datetime.now().isoformat(),
        }

        events.append(new_event)
        save_events(events)

        return (
            f"Successfully scheduled '{title}' "
            f"on {date_str} at {time_str}."
        )

    @mcp.tool()
    async def calendar_get_events() -> str:
        """Retrieve all scheduled calendar events."""

        events = load_events()

        if not events:
            return "No calendar events found."

        return json.dumps(events, indent=2)

    @mcp.tool()
    async def calendar_get_month_view(
        year: int,
        month: int,
    ) -> str:
        """Generate a text calendar for a specific month."""

        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12.")

        return calendar.month(year, month)