import sys
from pathlib import Path

import asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# MCP SERVER CONFIGURATION
# ============================================================

server_params = StdioServerParameters(
    command=sys.executable,
    args=[
        "-u",
        "-m",
        "src.ai_call_mcp.server",
    ],
    env=dict(
        **__import__("os").environ,
        PYTHONPATH=str(PROJECT_ROOT),
    ),
)


# ============================================================
# TEST
# ============================================================

async def test_whatsapp():

    print("\n========================================")
    print("       WHATSAPP MCP TOOL TEST")
    print("========================================\n")

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # ------------------------------------------------
            # 1. Initialize MCP
            # ------------------------------------------------

            print(">>> Initializing MCP server...")

            await session.initialize()

            print(">>> MCP server connected.\n")

            # ------------------------------------------------
            # 2. List MCP tools
            # ------------------------------------------------

            print(">>> Available MCP tools:")

            tools = await session.list_tools()

            for tool in tools.tools:
                print(f"   - {tool.name}")

            print()

            # ------------------------------------------------
            # 3. Check WhatsApp tool
            # ------------------------------------------------

            whatsapp_tool = next(
                (
                    tool
                    for tool in tools.tools
                    if tool.name == "whatsapp_send_message"
                ),
                None,
            )

            if whatsapp_tool is None:
                print("❌ whatsapp_send_message tool not found.")
                return

            print("✅ whatsapp_send_message tool found.\n")

            # ------------------------------------------------
            # 4. Get test phone number
            # ------------------------------------------------

            phone = input(
                "Enter test phone number with country code "
                "(example: 919876543210): "
            ).strip()

            if not phone:
                print("❌ Phone number cannot be empty.")
                return

            message = input(
                "Enter WhatsApp test message: "
            ).strip()

            if not message:
                message = (
                    "Hello! This is a test message "
                    "from the Vayvora AI Calling MCP."
                )

            # ------------------------------------------------
            # 5. Call MCP WhatsApp tool
            # ------------------------------------------------

            print("\n>>> Calling whatsapp_send_message...")

            result = await session.call_tool(
                "whatsapp_send_message",
                {
                    "phone": phone,
                    "message": message,
                    "session_id": "default",
                },
            )

            # ------------------------------------------------
            # 6. Display result
            # ------------------------------------------------

            print("\n>>> MCP TOOL RESULT:")

            for content in result.content:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(content)

            print("\n========================================")
            print("          TEST COMPLETED")
            print("========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    asyncio.run(test_whatsapp())