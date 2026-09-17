from typing import Any


class MCPAdapter:
    """
    Agent-side adapter for MCP tools.
    Connects the LangGraph agent to the FastMCP server.
    """

    def __init__(self, client: Any):
        self.client = client

    async def list_tools(self) -> list[Any]:
        return await self.client.list_tools()

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        if not tool_name:
            raise ValueError("MCP tool name cannot be empty.")

        arguments = arguments or {}

        return await self.client.call_tool(
            tool_name,
            arguments,
        )


class MCPTool:
    """
    Adapter that exposes an MCP tool through the Agent's
    standard ToolRegistry interface.
    """

    def __init__(
        self,
        adapter: MCPAdapter,
        tool_name: str,
    ):
        self.adapter = adapter
        self.tool_name = tool_name

    async def ainvoke(
        self,
        tool_input: dict[str, Any] | None = None,
    ) -> Any:
        return await self.adapter.call(
            self.tool_name,
            tool_input or {},
        )
