from typing import Any
from langgraph.graph import END, START, StateGraph

from src.agents.config import DEFAULT_AGENT_CONFIG, AgentConfig
from src.agents.edges.routing import (
    route_after_memory,
    route_after_rag,
    route_after_router,
    route_after_tool,
)
from src.agents.nodes.llm_node import LLMNode
from src.agents.nodes.memory_node import MemoryNode
from src.agents.nodes.rag_node import RAGNode
from src.agents.nodes.response_node import ResponseNode
from src.agents.nodes.router_node import RouterNode
from src.agents.nodes.tool_node import ToolNode
from src.agents.state import AgentState
from src.agents.tools.tool_selector import ToolSelector
from src.rag.embeddings.provider import EmbeddingProvider, get_embedding_provider


class AgentGraph:
    """
    LangGraph Workflow Assembly for Vayvora AI Voice Calling.
    Integrates the semantic router, conversational memory, hybrid RAG retriever,
    MCP tool executor, and speech LLM synthesis node.
    """

    def __init__(
        self,
        llm: Any,
        tool_registry: Any = None,
        memory_store: Any = None,
        retriever: Any = None,
        embedding_provider: EmbeddingProvider | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or DEFAULT_AGENT_CONFIG
        self.embedding_provider = embedding_provider or get_embedding_provider()

        # Semantic Router with Knowledge Base Resonance
        self.router_node = RouterNode(
            config=self.config,
            embedding_provider=self.embedding_provider,
        )

        self.memory_node = MemoryNode(memory_store=memory_store)
        self.rag_node = RAGNode(retriever=retriever)

        # Semantic Tool Selector for MCP tools
        tool_selector = ToolSelector(
            embedding_provider=self.embedding_provider,
            similarity_threshold=self.config.tool_similarity_threshold,
        )

        self.tool_node = ToolNode(
            tool_registry=tool_registry,
            tool_selector=tool_selector,
        )

        self.llm_node = LLMNode(llm=llm)
        self.response_node = ResponseNode()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self.router_node.run)
        workflow.add_node("memory", self.memory_node.run)
        workflow.add_node("rag", self.rag_node.run)
        workflow.add_node("tool", self.tool_node.run)
        workflow.add_node("llm", self.llm_node.run)
        workflow.add_node("response", self.response_node.run)

        workflow.add_edge(START, "router")

        # Routing conditional dispatch
        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "response": "response",
                "memory": "memory",
            },
        )

        # Memory inspection conditional dispatch
        workflow.add_conditional_edges(
            "memory",
            route_after_memory,
            {
                "llm": "llm",
                "rag": "rag",
                "tool": "tool",
            },
        )

        workflow.add_conditional_edges("rag", route_after_rag, {"llm": "llm"})
        workflow.add_conditional_edges(
            "tool",
            route_after_tool,
            {
                "llm": "llm",
                "response": "response",
            },
        )
        workflow.add_edge("llm", "response")
        workflow.add_edge("response", END)

        return workflow.compile()

    async def run(self, state: AgentState) -> AgentState:
        return await self.graph.ainvoke(state)
