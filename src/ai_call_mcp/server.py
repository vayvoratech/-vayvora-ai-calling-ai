from mcp.server.fastmcp import FastMCP
from tools.whatsapp import register_whatsapp_tools
from tools.mail import register_mail_tools
from tools.calendar import register_calendar_tools

# Initialize the FastMCP server
mcp = FastMCP("SelfHosted-Ecosystem-Server")

# Register tools from the tools folder
register_whatsapp_tools(mcp)
register_mail_tools(mcp)
register_calendar_tools(mcp)

if __name__ == "__main__":
    # Run the server using stdio transport
    mcp.run()