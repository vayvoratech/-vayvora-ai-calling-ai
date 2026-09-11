from typing import Any

from src.agents.state import AgentState


class RAGNode:
    """
    Retrieves grounded knowledge for the agent.

    RAG runs when:
    - the router requests knowledge retrieval
    - memory did not already satisfy the request
    - the user has provided a valid query

    The node stores both the structured RAG response and the
    final context that should be supplied to the LLM.
    """

    def __init__(self, retriever: Any = None):
        self.retriever = retriever

    async def run(self, state: AgentState) -> AgentState:
        # Memory already has relevant information.
        if state.get("memory_hit"):
            return {
                **state,
                "rag_required": False,
                "rag_context": "",
                "rag_response": None,
            }

        # RAG is not configured.
        if self.retriever is None:
            return {
                **state,
                "rag_required": False,
                "rag_context": "",
                "rag_response": None,
                "error": "RAG retriever is not configured.",
            }

        user_input = state.get("user_input", "").strip()

        if not user_input:
            return {
                **state,
                "rag_required": False,
                "rag_context": "",
                "rag_response": None,
            }

        try:
            rag_response = await self.retriever.search(
                query=user_input,
                tenant_id=state.get("tenant_id", "default"),
            )

            context = rag_response.context

            return {
                **state,
                "rag_required": rag_response.has_relevant_context,
                "rag_context": context,
                "rag_response": rag_response,
                "error": None,
            }

        except Exception as exc:
            return {
                **state,
                "rag_required": False,
                "rag_context": "",
                "rag_response": None,
                "error": f"RAG retrieval failed: {exc}",
            }