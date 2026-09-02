import os

import httpx
from mcp.server.fastmcp import FastMCP


def register_whatsapp_tools(mcp: FastMCP) -> None:
    """Register WhatsApp tools with the MCP server."""

    openwa_base_url = os.getenv(
        "OPENWA_BASE_URL",
        "http://localhost:2785",
    ).rstrip("/")

    openwa_api_key = os.getenv(
        "OPENWA_API_KEY",
        "",
    )

    @mcp.tool()
    async def whatsapp_send_message(
        phone: str,
        message: str,
        session_id: str = "default",
    ) -> str:
        """
        Send a WhatsApp message through the self-hosted
        OpenWA instance.
        """

        if not phone.strip():
            raise ValueError("Phone number cannot be empty.")

        if not message.strip():
            raise ValueError("Message cannot be empty.")

        url = (
            f"{openwa_base_url}/api/"
            f"{session_id}/send-message"
        )

        headers = {
            "Content-Type": "application/json",
            "x-api-key": openwa_api_key,
        }

        payload = {
            "phone": f"{phone}@c.us",
            "message": message,
        }

        try:
            async with httpx.AsyncClient(
                timeout=10.0
            ) as client:

                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                )

            if response.status_code != 200:
                raise RuntimeError(
                    "Failed to send WhatsApp message: "
                    f"{response.text}"
                )

            return (
                f"Successfully sent WhatsApp message "
                f"to {phone}."
            )

        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"WhatsApp service unavailable: {exc}"
            ) from exc