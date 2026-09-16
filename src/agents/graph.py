from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.edges.routing import (
    route_after_llm_router,
    route_after_memory,
    route_after_rag,
    route_after_router,
    route_after_tool,
)

from src.agents.nodes.llm_node import LLMNode
from src.agents.nodes.llm_router_node import LLMRouterNode
from src.agents.nodes.memory_node import MemoryNode
from src.agents.nodes.rag_node import RAGNode
from src.agents.nodes.response_node import ResponseNode
from src.agents.nodes.router_node import RouterNode
from src.agents.nodes.tool_node import ToolNode

from src.agents.state import AgentState


class AgentGraph:
    """
    Main Vayvora AI Agent orchestration graph.

    Architecture:

        User Input
            ↓
        Fast Router
            │
            ├── Direct ───────────────→ Response
            │
            ├── Clear RAG ──→ Memory ─→ RAG ─→ LLM
            │
            ├── Clear MCP ──→ Memory ─→ Tool ─→ LLM
            │
            └── Ambiguous
                    ↓
                LLM Router
                    │
                    ├── Direct ─→ Response
                    ├── RAG ─→ Memory ─→ RAG ─→ LLM
                    ├── MCP ─→ Memory ─→ Tool ─→ LLM
                    └── LLM ─→ Memory ─→ LLM
                                             ↓
                                          Response
    """

    def __init__(
        self,
        llm: Any,
        tool_registry: Any = None,
        memory_store: Any = None,
        retriever: Any = None,
    ) -> None:

        self.router_node = RouterNode()

        self.llm_router_node = LLMRouterNode(
            llm=llm,
        )

        self.memory_node = MemoryNode(
            memory_store=memory_store,
        )

        self.rag_node = RAGNode(
            retriever=retriever,
        )

        self.tool_node = ToolNode(
            tool_registry=tool_registry,
        )

        self.llm_node = LLMNode(
            llm=llm,
        )

        self.response_node = ResponseNode()

        self.graph = self._build_graph()

    def _build_graph(self):

        workflow = StateGraph(AgentState)

        # ─────────────────────────────────────────
        # NODES
        # ─────────────────────────────────────────

        workflow.add_node(
            "router",
            self.router_node.run,
        )

        workflow.add_node(
            "llm_router",
            self.llm_router_node.run,
        )

        workflow.add_node(
            "memory",
            self.memory_node.run,
        )

        workflow.add_node(
            "rag",
            self.rag_node.run,
        )

        workflow.add_node(
            "tool",
            self.tool_node.run,
        )

        workflow.add_node(
            "llm",
            self.llm_node.run,
        )

        workflow.add_node(
            "response",
            self.response_node.run,
        )

        # ─────────────────────────────────────────
        # START
        # ─────────────────────────────────────────

        workflow.add_edge(
            START,
            "router",
        )

        # ─────────────────────────────────────────
        # FAST ROUTER
        # ─────────────────────────────────────────

        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "response": "response",
                "llm_router": "llm_router",
                "memory": "memory",
            },
        )

        # ─────────────────────────────────────────
        # LLM FALLBACK ROUTER
        # ─────────────────────────────────────────

        workflow.add_conditional_edges(
            "llm_router",
            route_after_llm_router,
            {
                "response": "response",
                "memory": "memory",
            },
        )

        # ─────────────────────────────────────────
        # MEMORY
        # ─────────────────────────────────────────

        workflow.add_conditional_edges(
            "memory",
            route_after_memory,
            {
                "llm": "llm",
                "rag": "rag",
                "tool": "tool",
            },
        )

        # ─────────────────────────────────────────
        # RAG → LLM
        # ─────────────────────────────────────────

        workflow.add_conditional_edges(
            "rag",
            route_after_rag,
            {
                "llm": "llm",
            },
        )

        # ─────────────────────────────────────────
        # MCP → LLM
        # ─────────────────────────────────────────

        workflow.add_conditional_edges(
            "tool",
            route_after_tool,
            {
                "llm": "llm",
            },
        )

        # ─────────────────────────────────────────
        # LLM → RESPONSE
        # ─────────────────────────────────────────

        workflow.add_edge(
            "llm",
            "response",
        )

        # ─────────────────────────────────────────
        # RESPONSE → END
        # ─────────────────────────────────────────

        workflow.add_edge(
            "response",
            END,
        )

        return workflow.compile()

    async def run(
        self,
        state: AgentState,
    ) -> AgentState:
        """Execute one Agent turn."""

        return await self.graph.ainvoke(state)