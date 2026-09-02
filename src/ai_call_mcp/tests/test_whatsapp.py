import os
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENWA_BASE_URL = os.getenv("OPENWA_BASE_URL", "http://localhost:2785")
OPENWA_API_KEY = os.getenv("OPENWA_API_KEY", "")

def test_connection():
    print("--- 1. Testing OpenWA Health Check ---")
    try:
        response = httpx.get(f"{OPENWA_BASE_URL}/api/infra/health", timeout=5.0)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Failed to connect to OpenWA server: {e}")
        return

    print("\n--- 2. Fetching Active Sessions ---")
    headers = {"X-API-Key": OPENWA_API_KEY}
    
    try:
        response = httpx.get(f"{OPENWA_BASE_URL}/api/sessions", headers=headers, timeout=5.0)
        print(f"Status Code: {response.status_code}")
        sessions = response.json()
        print(f"Sessions Found: {sessions}")
        
        # If sessions exist, prompt to test sending a message
        if isinstance(sessions, list) and len(sessions) > 0:
            ready_session = next((s for s in sessions if s.get("status") == "ready"), sessions[0])
            session_id = ready_session.get("id")
            print(f"\nUsing Session ID: {session_id} (Status: {ready_session.get('status')})")
            
            if ready_session.get("status") == "ready":
                test_phone = input("Enter a test phone number with country code (e.g., 919876543210): ").strip()
                if test_phone:
                    send_test_message(session_id, test_phone, headers)
            else:
                print("Note: Session is not in 'ready' state yet. Scan the QR code via your dashboard/API before sending messages.")
        else:
            print("No sessions found. Create and start a session through your OpenWA dashboard first.")

    except Exception as e:
        print(f"Failed to fetch sessions: {e}")

def send_test_message(session_id: str, phone: str, headers: dict):
    print(f"\n--- 3. Sending Test Message to {phone} ---")
    url = f"{OPENWA_BASE_URL}/api/sessions/{session_id}/messages/send-text"
    
    payload = {
        "chatId": f"{phone}@c.us",
        "text": "Hello! This is a test message from my self-hosted OpenWA MCP client."
    }
    
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=10.0)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        if response.status_code in (200, 201):
            print("Success! WhatsApp message dispatched.")
        else:
            print("Message dispatch returned an error.")
    except Exception as e:
        print(f"Failed to send message request: {e}")

if __name__ == "__main__":
    test_connection()