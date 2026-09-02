import pytest

from src.agents.tools.mcp_adapter import MCPAdapter
from src.agents.tools.mcp_bootstrap import get_mcp_server_path
from src.agents.tools.mcp_client import MCPClient


@pytest.mark.anyio
async def test_mcp_tool_discovery():
    client = MCPClient(
        server_path=get_mcp_server_path(),
    )

    try:
        await client.connect()

        adapter = MCPAdapter(client)

        tools = await adapter.list_tools()

        tool_names = {
            tool.name
            for tool in tools
        }

        expected_tools = {
            "mail_send",
            "mail_read_recent",
            "calendar_add_event",
            "calendar_get_events",
            "calendar_get_month_view",
            "whatsapp_send_message",
        }

        assert expected_tools.issubset(tool_names)

    finally:
        await client.close()