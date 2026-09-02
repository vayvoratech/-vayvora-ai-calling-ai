import pytest

from src.agents.runtime import AgentRuntime


class FakeResponse:
    content = "Here are your calendar events."


class FakeLLM:
    async def ainvoke(self, messages):
        return FakeResponse()


@pytest.mark.anyio
async def test_agent_calendar_tool_flow():
    runtime = AgentRuntime(
        llm=FakeLLM(),
    )

    try:
        await runtime.start()

        result = await runtime.run(
            {
                "session_id": "test-session",
                "call_id": "test-call",
                "user_input": "What's on my calendar?",
                "messages": [],
            }
        )

        assert result["route"] == "mcp"
        assert result["tool_name"] == "calendar_get_events"
        assert result["tool_result"] is not None
        assert result["response"] == "Here are your calendar events."

    finally:
        await runtime.close()