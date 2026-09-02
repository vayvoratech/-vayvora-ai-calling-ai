from pathlib import Path

from src.agents.tools.mcp_adapter import MCPAdapter
from src.agents.tools.mcp_client import MCPClient
from src.agents.tools.mcp_registry import create_mcp_registry
from src.agents.tools.registry import ToolRegistry


def get_mcp_server_path() -> Path:
    """
    Resolve the Vayvora MCP server location.
    """

    project_root = Path(__file__).resolve().parents[3]

    return (
        project_root
        / "src"
        / "ai_call_mcp"
        / "server.py"
    )


async def create_agent_tool_registry() -> tuple[
    ToolRegistry,
    MCPClient,
]:
    """
    Start the MCP server, connect to it, discover its tools,
    and register those tools for the Agent.

    Returns:
        ToolRegistry: Agent-accessible tools.
        MCPClient: Persistent MCP connection.
    """

    server_path = get_mcp_server_path()

    if not server_path.exists():
        raise FileNotFoundError(
            f"MCP server not found: {server_path}"
        )

    client = MCPClient(
        server_path=server_path,
    )

    await client.connect()

    adapter = MCPAdapter(client)

    registry = await create_mcp_registry(
        adapter=adapter,
    )

    return registry, client