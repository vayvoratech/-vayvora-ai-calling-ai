from mcp.server.fastmcp import FastMCP

from src.ai_call_mcp.tools.calendar import register_calendar_tools
from src.ai_call_mcp.tools.mail import register_mail_tools
from src.ai_call_mcp.tools.whatsapp import register_whatsapp_tools


mcp = FastMCP("Vayvora-AI-MCP")


register_mail_tools(mcp)
register_calendar_tools(mcp)
register_whatsapp_tools(mcp)


if __name__ == "__main__":
    mcp.run()