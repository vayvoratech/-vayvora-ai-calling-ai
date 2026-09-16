import sys
from pathlib import Path
import asyncio
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_call_mcp.tools.calendar import (
    load_events,
    save_events,
    EVENTS_FILE,
)


async def test_calendar():
    print("\n========== CALENDAR TEST ==========\n")

    # --------------------------------------------------
    # 1. Show events file location
    # --------------------------------------------------

    print(f"Events file: {EVENTS_FILE}")

    # --------------------------------------------------
    # 2. Load existing events
    # --------------------------------------------------

    events = load_events()

    print("\nExisting events:")
    print(json.dumps(events, indent=4))

    # --------------------------------------------------
    # 3. Add test event
    # --------------------------------------------------

    new_event = {
        "id": len(events) + 1,
        "title": "Test Calendar Event",
        "date": "2026-09-20",
        "time": "10:30",
        "description": "Calendar tool test event",
    }

    events.append(new_event)

    save_events(events)

    print("\nEvent added successfully:")
    print(json.dumps(new_event, indent=4))

    # --------------------------------------------------
    # 4. Load again and verify
    # --------------------------------------------------

    updated_events = load_events()

    print("\nEvents after saving:")
    print(json.dumps(updated_events, indent=4))

    # --------------------------------------------------
    # 5. Verify
    # --------------------------------------------------

    found = any(
        event.get("title") == "Test Calendar Event"
        for event in updated_events
    )

    if found:
        print("\n✅ CALENDAR TEST PASSED")
    else:
        print("\n❌ CALENDAR TEST FAILED")


if __name__ == "__main__":
    asyncio.run(test_calendar())