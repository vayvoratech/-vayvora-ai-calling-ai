import calendar
import json
import os
from datetime import datetime

EVENTS_FILE = "events.json"

def load_events():
    """Load events from the local JSON storage file."""
    if not os.path.exists(EVENTS_FILE):
        return []
    try:
        with open(EVENTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_events(events):
    """Save events to the local JSON storage file."""
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=4)

def add_event(title: str, date_str: str, time_str: str, description: str = "") -> str:
    """
    Add a new calendar event.
    :param title: Event title/name
    :param date_str: Date in 'YYYY-MM-DD' format
    :param time_str: Time in 'HH:MM' format
    :param description: Optional details or notes
    """
    events = load_events()
    new_event = {
        "id": len(events) + 1,
        "title": title,
        "date": date_str,
        "time": time_str,
        "description": description,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    events.append(new_event)
    save_events(events)
    return f"Success: Scheduled '{title}' on {date_str} at {time_str}."

def get_events() -> list:
    """Retrieve all scheduled calendar events."""
    return load_events()

def get_month_view(year: int, month: int) -> str:
    """Generate a visual text-based calendar grid for a specific month and year."""
    return calendar.month(year, month)