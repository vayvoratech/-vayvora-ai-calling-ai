"""Calendar provider implementations and persistence for external MCP tool execution.

Provides FileCalendarProvider for persistent calendar event management via JSON,
and MockCalendarProvider for deterministic unit testing. Reuses and adapts
event persistence and month view capabilities from the legacy MCP calendar tool.
"""

from abc import ABC, abstractmethod
import calendar
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.logging import get_logger

logger = get_logger("tools.calendar_provider")


class CalendarEventRecord(BaseModel):
    """Structured calendar event representation."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Meeting title or consultation topic")
    date: str = Field(..., description="Event date in YYYY-MM-DD or formatted string")
    time: str = Field(..., description="Event time in HH:MM or formatted slot string")
    slot: str = Field(default="", description="Combined slot label, e.g. 'Tomorrow 10:00 AM'")
    description: str = Field(default="", description="Meeting description or notes")
    attendee_name: Optional[str] = Field(default=None, description="Attendee full name")
    attendee_email: Optional[str] = Field(default=None, description="Attendee email address")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp")


class CalendarProvider(ABC):
    """Abstract interface for calendar scheduling and availability operations."""

    @abstractmethod
    def add_event(
        self,
        title: str,
        slot: Optional[str] = None,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None,
        description: str = "",
        attendee_name: Optional[str] = None,
        attendee_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Book a calendar event and return structured confirmation with event_id."""
        pass

    @abstractmethod
    def get_events(self) -> List[Dict[str, Any]]:
        """Retrieve all currently booked calendar appointments."""
        pass

    @abstractmethod
    def get_available_slots(
        self,
        preferred_date: Optional[str] = None,
        meeting_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query available consultation or meeting slots."""
        pass

    @abstractmethod
    def get_month_view(self, year: int, month: int) -> str:
        """Generate text calendar representation for a given month."""
        pass


class FileCalendarProvider(CalendarProvider):
    """File-backed calendar provider persisting events to JSON file."""

    DEFAULT_DAILY_SLOTS = [
        "10:00 AM",
        "11:30 AM",
        "02:00 PM",
        "03:30 PM",
        "05:00 PM",
        "06:00 PM",
    ]

    def __init__(
        self,
        settings: Optional[Settings] = None,
        events_path: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        file_path_str = events_path or getattr(self.settings, "calendar_events_path", "data/events.json")
        self.file_path = Path(file_path_str).resolve()
        self._lock = threading.Lock()
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Create parent directory and file if not present."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.file_path.exists():
                with self.file_path.open("w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
        except Exception as exc:
            logger.warning("Could not initialize calendar file at %s: %s", self.file_path, exc)

    def _load_events(self) -> List[Dict[str, Any]]:
        """Thread-safe read of events from file."""
        if not self.file_path.exists():
            return []
        with self._lock:
            try:
                with self.file_path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
                return data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Error reading events from %s: %s", self.file_path, exc)
                return []

    def _save_events(self, events: List[Dict[str, Any]]) -> None:
        """Thread-safe write of events to file."""
        with self._lock:
            try:
                with self.file_path.open("w", encoding="utf-8") as file:
                    json.dump(events, file, indent=2)
            except OSError as exc:
                logger.error("Error saving events to %s: %s", self.file_path, exc)
                raise

    def add_event(
        self,
        title: str,
        slot: Optional[str] = None,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None,
        description: str = "",
        attendee_name: Optional[str] = None,
        attendee_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Book an appointment, storing to JSON file and returning verified event_id."""
        if not title or not title.strip():
            title = "Consultation Meeting"

        # Determine date and time
        effective_date = date_str or ""
        effective_time = time_str or ""
        effective_slot = slot or ""

        if not effective_slot and effective_date and effective_time:
            effective_slot = f"{effective_date} {effective_time}"

        if not effective_slot and not (effective_date and effective_time):
            raise ValueError("Calendar Safety Violation: A valid slot or date/time must be specified.")

        # Validate date_str and time_str format if explicitly provided as YYYY-MM-DD / HH:MM
        if date_str and time_str:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                datetime.strptime(time_str, "%H:%M")
            except ValueError:
                # If they don't match strict format, ensure they are non-empty strings
                if not date_str.strip() or not time_str.strip():
                    raise ValueError("Invalid date or time format.")

        events = self._load_events()
        event_num = len(events) + 1
        event_id = f"evt_cal_{event_num}_{uuid.uuid4().hex[:6]}"

        new_record: Dict[str, Any] = {
            "id": event_id,
            "event_id": event_id,
            "numeric_id": event_num,
            "title": title.strip(),
            "date": effective_date or (effective_slot.split()[0] if effective_slot else "Upcoming"),
            "time": effective_time or (" ".join(effective_slot.split()[1:]) if " " in effective_slot else effective_slot),
            "slot": effective_slot,
            "description": description.strip(),
            "attendee_name": attendee_name,
            "attendee_email": attendee_email,
            "created_at": datetime.now().isoformat(),
            "status": "confirmed",
        }

        events.append(new_record)
        self._save_events(events)

        logger.info("Calendar event successfully created with ID '%s': %s", event_id, title)
        return {
            "event_id": event_id,
            "id": event_id,
            "status": "confirmed",
            "title": title.strip(),
            "slot": effective_slot,
            "date": new_record["date"],
            "time": new_record["time"],
            "message": f"Successfully scheduled '{title}' on {new_record['date']} at {new_record['time']}.",
        }

    def get_events(self) -> List[Dict[str, Any]]:
        """Retrieve all scheduled events from persistence."""
        return self._load_events()

    def get_available_slots(
        self,
        preferred_date: Optional[str] = None,
        meeting_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute available meeting slots, checking against existing bookings."""
        target_day = "Tomorrow"
        if preferred_date:
            target_day = preferred_date.strip().title()

        events = self._load_events()
        booked_times = {
            e.get("time", "").strip().upper()
            for e in events
            if e.get("date", "").strip().lower() == target_day.lower() or target_day.lower() in e.get("slot", "").lower()
        }

        available = [
            f"{target_day} {slot}"
            for slot in self.DEFAULT_DAILY_SLOTS
            if slot.upper() not in booked_times
        ]

        if not available:
            # Fallback next business day slots
            available = [
                f"Next Monday 10:00 AM",
                f"Next Monday 02:00 PM",
                f"Next Tuesday 11:00 AM",
            ]

        logger.info("Found %d available calendar slots for date '%s'", len(available), target_day)
        return {
            "slots": available,
            "count": len(available),
            "preferred_date": target_day,
            "meeting_type": meeting_type or "standard_consultation",
        }

    def get_month_view(self, year: int, month: int) -> str:
        """Generate text calendar for month."""
        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12.")
        return calendar.month(year, month)


class MockCalendarProvider(CalendarProvider):
    """In-memory mock calendar provider for isolated test runs."""

    def __init__(
        self,
        force_failure: bool = False,
        force_verification_failure: bool = False,
    ) -> None:
        self.force_failure = force_failure
        self.force_verification_failure = force_verification_failure
        self.events: List[Dict[str, Any]] = []

    def add_event(
        self,
        title: str,
        slot: Optional[str] = None,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None,
        description: str = "",
        attendee_name: Optional[str] = None,
        attendee_email: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.force_failure:
            raise RuntimeError("Simulated calendar provider outage")

        if self.force_verification_failure:
            # Server returns confirmation without external event_id
            return {"status": "confirmed", "slot": slot or "Tomorrow 10:00 AM"}

        effective_slot = slot or (f"{date_str} {time_str}" if date_str and time_str else "Tomorrow 10:00 AM")
        event_id = f"evt_mock_{len(self.events) + 1}_{uuid.uuid4().hex[:6]}"
        record = {
            "id": event_id,
            "event_id": event_id,
            "title": title,
            "slot": effective_slot,
            "date": date_str or "Tomorrow",
            "time": time_str or "10:00 AM",
            "description": description,
            "status": "confirmed",
        }
        self.events.append(record)
        return {
            "event_id": event_id,
            "id": event_id,
            "status": "confirmed",
            "title": title,
            "slot": effective_slot,
            "message": f"Successfully scheduled '{title}' for {effective_slot}.",
        }

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def get_available_slots(
        self,
        preferred_date: Optional[str] = None,
        meeting_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.force_failure:
            return {"error": "Calendar service unavailable"}
        return {
            "slots": ["Tomorrow 10:00 AM", "Tomorrow 02:00 PM", "Next Monday 11:00 AM"],
            "count": 3,
        }

    def get_month_view(self, year: int, month: int) -> str:
        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12.")
        return calendar.month(year, month)
