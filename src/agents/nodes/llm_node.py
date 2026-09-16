from typing import Any

from src.agents.prompts.system import SYSTEM_PROMPT
from src.agents.state import AgentState


class LLMNode:
    """
    Single LLM execution layer for Vayvora AI.

    The LLM is called after the agent has gathered the required
    information from:

        Memory
        RAG
        MCP / Tools

    Flow:

        Memory / RAG / MCP
                ↓
             LLMNode
                ↓
        Final spoken response
    """

    def __init__(self, llm: Any):
        self.llm = llm

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Safely extract plain text from str, list of chunks/dicts, or custom objects."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        if isinstance(content, dict):
            for key in ("text", "content", "message"):
                if key in content and isinstance(content[key], str):
                    return content[key].strip()
            return str(content).strip()
        return str(content).strip()

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        # ---------------------------------------------------------
        # Empty input
        # ---------------------------------------------------------

        if not user_input:
            return {
                **state,
                "response": "I'm sorry, I didn't hear anything.",
                "is_complete": True,
            }

        # ---------------------------------------------------------
        # Existing conversation messages
        # ---------------------------------------------------------

        messages = list(state.get("messages", []))

        # ---------------------------------------------------------
        # System prompt
        # ---------------------------------------------------------

        if not any(
            message.get("role") == "system"
            for message in messages
        ):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
            )

        # ---------------------------------------------------------
        # Memory context
        # ---------------------------------------------------------

        memory_context = state.get("memory_context", [])

        if memory_context:
            memory_text = "\n".join(
                str(item)
                for item in memory_context
            )

            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Relevant conversation memory:\n\n"
                        f"{memory_text}\n\n"
                        "Use this memory when it is relevant to "
                        "the user's current request."
                    ),
                }
            )

        # ---------------------------------------------------------
        # RAG context
        # ---------------------------------------------------------

        rag_context = state.get("rag_context", "")

        if rag_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "GROUNDING INSTRUCTION:\n"
                        "The following information was retrieved "
                        "from the Vayvora knowledge base.\n\n"
                        "Use this information as the source of truth "
                        "for company, service, pricing, policy, "
                        "process, support, portfolio, and other "
                        "knowledge-base questions.\n\n"
                        "Do not invent, assume, or fabricate facts "
                        "that are not supported by the retrieved "
                        "knowledge.\n\n"
                        "If the retrieved knowledge does not contain "
                        "the requested information, clearly state "
                        "that the information is not available in "
                        "the current knowledge base.\n\n"
                        "RETRIEVED KNOWLEDGE:\n"
                        f"{rag_context}"
                    ),
                }
            )

        # ---------------------------------------------------------
        # MCP / Tool result
        # ---------------------------------------------------------

        tool_result = state.get("tool_result")

        if tool_result is not None:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Result from the requested external "
                        "action or live data source:\n\n"
                        f"{tool_result}\n\n"
                        "Use this result when answering the user's "
                        "request. Do not claim an external action "
                        "succeeded unless the tool result supports it."
                    ),
                }
            )

        # ---------------------------------------------------------
        # Current user message
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # LLM execution
        # ---------------------------------------------------------

        try:
            response = await self.llm.ainvoke(messages)

            raw_content = getattr(response, "content", response)
            content = self._extract_text(raw_content)

            if not content:
                content = (
                    "I'm sorry, I wasn't able to generate a response."
                )

            # -----------------------------------------------------
            # Conversation history
            # -----------------------------------------------------

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
                "response": (
                    "I'm sorry, I'm having trouble processing "
                    "that right now."
                ),
                "is_complete": True,
                "error": str(exc),
            }