from typing import Any

from src.agents.tools.mcp_adapter import MCPAdapter, MCPTool
from src.agents.tools.registry import ToolRegistry


async def register_mcp_tools(
    registry: ToolRegistry,
    adapter: MCPAdapter,
) -> list[str]:
    """
    Discover tools exposed by the MCP server and register them
    in the Agent ToolRegistry.

    Returns:
        Names of successfully registered MCP tools.
    """

    tools = await adapter.list_tools()

    registered: list[str] = []

    for tool in tools:
        tool_name = getattr(tool, "name", None)

        if not tool_name:
            continue

        registry.register(
            tool_name,
            MCPTool(
                adapter=adapter,
                tool_name=tool_name,
            ),
        )

        registered.append(tool_name)

    return registered


async def create_mcp_registry(
    adapter: MCPAdapter,
) -> ToolRegistry:
    """
    Create a ToolRegistry and automatically populate it with
    tools exposed by the MCP server.
    """

    registry = ToolRegistry()

    await register_mcp_tools(
        registry=registry,
        adapter=adapter,
    )

    return registry