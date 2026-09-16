import sys
import os

# Fix for Windows: prevent stdin/stdout CRLF translations from corrupting JSON-RPC
if sys.platform == "win32":
    import msvcrt
    try:
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    except Exception:
        pass

import logging
from mcp.server.fastmcp import FastMCP

from src.ai_call_mcp.tools.calendar import register_calendar_tools
from src.ai_call_mcp.tools.mail import register_mail_tools
from src.ai_call_mcp.tools.whatsapp import register_whatsapp_tools

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

mcp = FastMCP("Vayvora-AI-MCP")

register_mail_tools(mcp)
register_calendar_tools(mcp)
register_whatsapp_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport="stdio")