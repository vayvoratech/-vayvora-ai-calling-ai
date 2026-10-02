import asyncio
import json
import websockets

async def test_session():
    uri = "wss://vayvora-ai-calling-ai-pzf9.onrender.com/media-stream?session_id=manual_test_01&domain=edusaas&direction=inbound"
    print(f"Connecting to: {uri}")

    try:
        async with websockets.connect(
            uri,
            open_timeout=30,     # Give cloud proxy 30s to complete TLS/HTTP upgrade
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print("Connected to Render Voice WebSocket!")

            start_payload = {
                "event": "start",
                "start": {
                    "streamSid": "test_stream_01",
                    "callSid": "test_call_01",
                    "customParameters": {
                        "domain": "edusaas",
                        "direction": "inbound"
                    }
                }
            }
            await ws.send(json.dumps(start_payload))
            print("Sent start handshake event.")

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                print(f"Received message: {msg}")
            except asyncio.TimeoutError:
                print("Connection established and held open.")

    except Exception as e:
        print(f"Connection test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_session())