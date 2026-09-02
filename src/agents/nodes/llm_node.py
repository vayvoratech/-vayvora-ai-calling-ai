from typing import Any

from src.agents.prompts.system import SYSTEM_PROMPT
from src.agents.state import AgentState


class LLMNode:
    """
    Single LLM execution layer for Vayvora AI.

    The LLM is called only after the Agent has gathered the
    required context from memory, RAG, or MCP.

    Flow:

        Memory / RAG / MCP
                ↓
             LLMNode
                ↓
        spoken response
    """

    def __init__(self, llm: Any):
        self.llm = llm

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        if not user_input:
            return {
                **state,
                "response": "I'm sorry, I didn't hear anything.",
                "is_complete": True,
            }

        messages = list(state.get("messages", []))

        # Add system instructions only once.
        if not any(message.get("role") == "system" for message in messages):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
            )

        # Add the current user message if it isn't already present.
        if not any(
            message.get("role") == "user"
            and message.get("content") == user_input
            for message in messages
        ):
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

        # Add relevant Redis memory.
        memory_context = state.get("memory_context", [])

        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant conversation memory:\n"
                        + "\n".join(str(item) for item in memory_context)
                    ),
                }
            )

        # Add RAG context only when available.
        rag_context = state.get("rag_context", [])

        if rag_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant knowledge retrieved from the "
                        "knowledge base:\n"
                        + "\n".join(str(item) for item in rag_context)
                    ),
                }
            )

        # Add MCP/tool result only when available.
        tool_result = state.get("tool_result")

        if tool_result is not None:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Result from the requested external action:\n"
                        f"{tool_result}"
                    ),
                }
            )

        try:
            response = await self.llm.ainvoke(messages)

            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            content = content.strip()

            updated_messages = messages + [
                {
                    "role": "assistant",
                    "content": content,
                }
            ]

            return {
                **state,
                "messages": updated_messages,
                "response": content,
                "is_complete": True,
                "error": None,
            }

        except Exception as exc:
            return {
                **state,
                "error": str(exc),
                "response": (
                    "I'm sorry, I'm having trouble processing "
                    "that right now."
                ),
                "is_complete": True,
            }