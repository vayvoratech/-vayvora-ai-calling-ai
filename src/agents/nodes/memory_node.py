from typing import Any

from src.agents.state import AgentState


class MemoryNode:
    """
    Retrieves relevant conversation memory before expensive
    RAG, MCP, or LLM operations.

    The node is intentionally storage-agnostic.
    Redis will be connected through a separate memory layer.
    """

    def __init__(self, memory_store: Any = None):
        self.memory_store = memory_store

    async def run(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        # No memory backend configured.
        if self.memory_store is None:
            return {
                **state,
                "memory_context": [],
                "memory_hit": False,
            }

        if not user_input:
            return {
                **state,
                "memory_context": [],
                "memory_hit": False,
            }

        try:
            memories = await self.memory_store.search(
                session_id=state.get("session_id", ""),
                query=user_input,
            )

            if not memories:
                return {
                    **state,
                    "memory_context": [],
                    "memory_hit": False,
                }

            return {
                **state,
                "memory_context": list(memories),
                "memory_hit": True,
            }

        except Exception as exc:
            # Memory failure must not break the voice call.
            return {
                **state,
                "memory_context": [],
                "memory_hit": False,
                "error": str(exc),
            }