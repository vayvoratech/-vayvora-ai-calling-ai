from typing import Any

from src.agents.state import AgentState


class RAGNode:
    """
    Retrieves relevant knowledge for the agent.

    RAG only runs when the router determines that external
    knowledge is required and memory does not already contain
    sufficient context.
    """

    def __init__(self, retriever: Any = None):
        self.retriever = retriever

    async def run(self, state: AgentState) -> AgentState:
        # Memory already has relevant information.
        # Avoid an unnecessary RAG call.
        if state.get("memory_hit"):
            return {
                **state,
                "rag_required": False,
                "rag_context": [],
            }

        if self.retriever is None:
            return {
                **state,
                "rag_required": False,
                "rag_context": [],
                "error": "RAG retriever is not configured.",
            }

        user_input = state.get("user_input", "").strip()

        if not user_input:
            return {
                **state,
                "rag_required": False,
                "rag_context": [],
            }

        try:
            results = await self.retriever.search(
                query=user_input
            )

            context = list(results) if results else []

            return {
                **state,
                "rag_required": bool(context),
                "rag_context": context,
            }

        except Exception as exc:
            # RAG failure should not terminate the call.
            return {
                **state,
                "rag_required": False,
                "rag_context": [],
                "error": str(exc),
            }