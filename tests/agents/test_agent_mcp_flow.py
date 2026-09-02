import pytest

from src.agents.runtime import AgentRuntime


class FakeResponse:
    content = "The calendar event was created successfully."


class FakeLLM:
    async def ainvoke(self, messages):
        return FakeResponse()


@pytest.mark.anyio
async def test_agent_mcp_flow():
    runtime = AgentRuntime(
        llm=FakeLLM(),
    )

    try:
        await runtime.start()

        assert runtime.agent is not None
        assert runtime.mcp_client is not None

        result = await runtime.agent.tool_node.tool_registry.get(
            "calendar_get_events"
        ).ainvoke({})

        assert result is not None

    finally:
        await runtime.close()