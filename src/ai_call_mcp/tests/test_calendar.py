from datetime import datetime
from tools.calendar import add_event, get_events, get_month_view

def run_tests():
    print("--- 1. Testing Month View Layout ---")
    now = datetime.now()
    calendar_grid = get_month_view(now.year, now.month)
    print(calendar_grid)

    print("--- 2. Testing Event Addition ---")
    response = add_event(
        title="MCP Architecture Sync",
        date_str=now.strftime("%Y-%m-%d"),
        time_str="15:30",
        description="Verify local native tools configuration."
    )
    print(response)

    print("\n--- 3. Testing Event Retrieval ---")
    events = get_events()
    print(f"Total events found in local storage: {len(events)}")
    for event in events:
        print(f"[{event['id']}] {event['title']} on {event['date']} at {event['time']}")
        if event['description']:
            print(f"    Note: {event['description']}")

if __name__ == "__main__":
    run_tests()