from contextlib import AsyncExitStack
import os
from pathlib import Path
import sys
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
        python_executable: str | None = None,
    ) -> None:
        self.server_path = Path(server_path).resolve()
        self.python_executable = python_executable or sys.executable or "python"

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

        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        server_parameters = StdioServerParameters(
            command=self.python_executable,
            args=[
                "-u",
                "-m",
                "src.ai_call_mcp.server",
            ],
            cwd=str(project_root),
            env=env,
        )

        # 1. Initialize and enter the AsyncExitStack FIRST
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        # 2. Enter stdio_client using the active exit stack
        read_stream, write_stream = (
            await self._exit_stack.enter_async_context(
                stdio_client(server_parameters)
            )
        )

        # 3. Enter ClientSession using the active exit stack
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
            )
        )

        # 4. Perform initialization handshake
        await self._session.initialize()

    async def list_tools(self) -> list[Any]:
        await self._ensure_connected()
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        await self._ensure_connected()
        result = await self._session.call_tool(
            tool_name,
            arguments or {},
        )
        return self._extract_result(result)

    async def close(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
        self._exit_stack = None
        self._session = None

    async def _ensure_connected(self) -> None:
        if self._session is None:
            await self.connect()

    @staticmethod
    def _extract_result(result: Any) -> Any:
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