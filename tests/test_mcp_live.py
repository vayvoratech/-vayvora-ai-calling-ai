"""Optional live integration test with an external MCP server.

This test is automatically skipped if a live MCP server is not reachable
on the configured MCP_SERVER_URL endpoint.
"""

import pytest
import httpx
from src.config import Settings
from src.core.types import ToolCallRequest
from src.tools.mcp_client import HttpMCPToolProvider


def is_mcp_reachable(url: str) -> bool:
    """Synchronous check if an MCP server responds at the given endpoint."""
    if not url or "localhost" not in url and "127.0.0.1" not in url:
        return False
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(url)
            return resp.status_code in (200, 404, 405)
    except Exception:
        return False


settings = Settings()
MCP_LIVE_AVAILABLE = is_mcp_reachable(settings.mcp_server_url)


@pytest.mark.skipif(
    not MCP_LIVE_AVAILABLE,
    reason="No live MCP server is reachable at MCP_SERVER_URL. Skipping live integration test.",
)
@pytest.mark.asyncio
async def test_live_mcp_server_invocation():
    """Live integration test: execute tool call against live HTTP MCP server."""
    provider = HttpMCPToolProvider(settings=settings)
    req = ToolCallRequest(
        tool_name="find_available_slots",
        arguments={"date": "tomorrow"},
        call_id="live-test-call-01",
    )
    result = await provider.execute_tool(req)
    assert result.tool_name == "find_available_slots"
