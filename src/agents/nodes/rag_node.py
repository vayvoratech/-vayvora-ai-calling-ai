from typing import Any
from src.agents.state import AgentState


class RAGNode:
    """
    Retrieves grounded company knowledge for the agent.

    Runs when:
      - The semantic router selected 'rag' based on high vector proximity to company knowledge
      - Memory did not already satisfy the request
      - The query contains valid semantic intent
    """

    def __init__(self, retriever: Any = None):
        self.retriever = retriever

    async def run(self, state: AgentState) -> AgentState:
        if state.get("memory_hit"):
            return {
                **state,
                "rag_required": False,
                "rag_context": "",
                "rag_response": None,
            }

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

        from src.agents.nodes.router_node import RouterNode
        if RouterNode.is_general_query(user_input):
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
