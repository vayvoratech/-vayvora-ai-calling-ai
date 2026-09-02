import pytest

from src.agents.graph import AgentGraph
from src.agents.tools.registry import ToolRegistry


class FakeResponse:
    content = "I found the customer information successfully."


class FakeLLM:
    async def ainvoke(self, messages):
        return FakeResponse()


class FakeTool:
    async def ainvoke(self, tool_input):
        return {
            "customer_name": "Test Customer",
            "status": "active",
        }


@pytest.mark.anyio
async def test_agent_mcp_tool_flow():
    llm = FakeLLM()
    registry = ToolRegistry()

    registry.register("get_customer", FakeTool())

    agent = AgentGraph(
        llm=llm,
        tool_registry=registry,
    )

    state = {
        "call_id": "test-call",
        "session_id": "test-session",
        "user_input": "Check my customer information",
        "messages": [],
        "route": "mcp",
        "tool_name": "get_customer",
        "tool_input": {},
    }

    result = await agent.run(state)

    assert result["tool_result"] == {
        "customer_name": "Test Customer",
        "status": "active",
    }

    assert result["response"] == "I found the customer information successfully."
    assert result["is_complete"] is True