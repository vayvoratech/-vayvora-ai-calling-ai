import pytest

from src.agents.graph import AgentGraph
from src.agents.tools.registry import ToolRegistry


class FakeResponse:
    content = "Hello from Vayvora AI."


class FakeLLM:
    async def ainvoke(self, messages):
        return FakeResponse()


@pytest.mark.anyio
async def test_direct_route():
    llm = FakeLLM()
    registry = ToolRegistry()

    agent = AgentGraph(
        llm=llm,
        tool_registry=registry,
    )

    state = {
        "call_id": "test-call",
        "session_id": "test-session",
        "user_input": "Hello",
        "messages": [],
    }

    result = await agent.run(state)

    assert result["route"] == "direct"
    assert result["is_complete"] is True