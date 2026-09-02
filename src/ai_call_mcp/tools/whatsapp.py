import os
import httpx
from mcp.server.fastmcp import FastMCP

def register_whatsapp_tools(mcp: FastMCP):
    OPENWA_BASE_URL = os.getenv("OPENWA_BASE_URL", "http://localhost:2785")
    OPENWA_API_KEY = os.getenv("OPENWA_API_KEY", "")

    @mcp.tool()
    async def whatsapp_send_message(phone: str, message: str, session_id: str = "default") -> str:
        """Send a WhatsApp message via your self-hosted OpenWA instance."""
        url = f"{OPENWA_BASE_URL}/api/{session_id}/send-message"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": OPENWA_API_KEY
        }
        payload = {
            "phone": f"{phone}@c.us",
            "message": message
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to send WhatsApp message: {response.text}")
                
            return f"Successfully sent WhatsApp message to {phone}."