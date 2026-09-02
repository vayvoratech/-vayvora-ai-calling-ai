from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    Persistent stdio client for the Vayvora AI MCP server.
    """

    def __init__(
        self,
        server_path: str | Path,
        python_executable: str = "python",
    ) -> None:
        self.server_path = Path(server_path).resolve()
        self.python_executable = python_executable

        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        """Start the MCP server and establish a persistent session."""

        if self._session is not None:
            return

        if not self.server_path.exists():
            raise FileNotFoundError(
                f"MCP server not found: {self.server_path}"
            )

        project_root = self.server_path.parents[2]

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        server_parameters = StdioServerParameters(
            command=self.python_executable,
            args=[
                "-m",
                "src.ai_call_mcp.server",
            ],
            cwd=str(project_root),
            env=None,
        )

        read_stream, write_stream = (
            await self._exit_stack.enter_async_context(
                stdio_client(server_parameters)
            )
        )

        self._session = await self._exit_stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
            )
        )

        await self._session.initialize()

    async def list_tools(self) -> list[Any]:
        """Return tools exposed by the MCP server."""

        await self._ensure_connected()

        result = await self._session.list_tools()

        return result.tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Execute an MCP tool."""

        await self._ensure_connected()

        result = await self._session.call_tool(
            tool_name,
            arguments or {},
        )

        return self._extract_result(result)

    async def close(self) -> None:
        """Close the MCP connection."""

        if self._exit_stack is not None:
            await self._exit_stack.aclose()

        self._exit_stack = None
        self._session = None

    async def _ensure_connected(self) -> None:
        if self._session is None:
            await self.connect()

    @staticmethod
    def _extract_result(result: Any) -> Any:
        """Extract useful content from an MCP tool result."""

        if getattr(result, "isError", False):
            raise RuntimeError(
                f"MCP tool execution failed: {result}"
            )

        content = getattr(result, "content", None)

        if not content:
            return result

        extracted = []

        for item in content:
            text = getattr(item, "text", None)

            if text is not None:
                extracted.append(text)
            else:
                extracted.append(item)

        if len(extracted) == 1:
            return extracted[0]

        return extracted