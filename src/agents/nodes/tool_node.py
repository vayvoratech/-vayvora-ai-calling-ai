from typing import Any

from src.agents.state import AgentState
from src.agents.tools.tool_selector import ToolSelector


class ToolNode:
    """
    MCP / external tool execution node.

    Responsibilities:
    1. Select an MCP tool locally.
    2. Extract simple tool arguments locally.
    3. Execute the selected tool.
    4. Store the result in AgentState.
    5. Pass the result to the single LLM.

    No LLM call is made for tool selection or
    simple argument extraction.
    """

    def __init__(
        self,
        tool_registry: Any = None,
        tool_selector: ToolSelector | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.tool_selector = tool_selector or ToolSelector()

    async def run(self, state: AgentState) -> AgentState:
        if self.tool_registry is None:
            return {
                **state,
                "tool_required": False,
                "error": "Tool registry is not configured.",
            }

        user_input = state.get("user_input", "").strip()

        tool_name = state.get("tool_name")
        tool_input = state.get("tool_input", {})

        if not tool_name:
            selection = self.tool_selector.select(user_input)

            if selection is None:
                return {
                    **state,
                    "tool_required": False,
                    "error": (
                        "Unable to determine the required MCP tool."
                    ),
                }

            tool_name = selection.tool_name
            tool_input = selection.tool_input

        tool = self.tool_registry.get(tool_name)

        if tool is None:
            return {
                **state,
                "tool_required": False,
                "error": f"Tool not found: {tool_name}",
            }

        try:
            result = await tool.ainvoke(tool_input)

            messages = list(state.get("messages", []))

            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result),
                }
            )

            return {
                **state,
                "messages": messages,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": result,
                "tool_required": False,
                "error": None,
            }

        except Exception as exc:
            return {
                **state,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_required": False,
                "error": str(exc),
            }