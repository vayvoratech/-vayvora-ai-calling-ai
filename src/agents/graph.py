from typing import Any

from langgraph.graph import END, START, StateGraph

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


class AgentGraph:
    """
    Main Vayvora AI Agent orchestration graph.

    Flow:

        Input
          ↓
        Fast Local Router
          ↓
        Memory
          ↓
        ┌──────────┬──────────┬──────────┐
        │          │          │          │
      Direct      LLM        RAG        MCP
        │          │          │          │
        │          │          └──→ LLM   │
        │          │                     │
        │          └─────────────────────┘
        │
        └────────────→ Response
    """

    def __init__(
        self,
        llm: Any,
        tool_registry: Any = None,
        memory_store: Any = None,
        retriever: Any = None,
    ) -> None:
        self.router_node = RouterNode()

        self.memory_node = MemoryNode(
            memory_store=memory_store
        )

        self.rag_node = RAGNode(
            retriever=retriever
        )

        self.tool_node = ToolNode(
            tool_registry=tool_registry
        )

        self.llm_node = LLMNode(
            llm=llm
        )

        self.response_node = ResponseNode()

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node(
            "router",
            self.router_node.run,
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

        # Start
        workflow.add_edge(
            START,
            "router",
        )

        # Router
        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "response": "response",
                "llm": "memory",
                "rag": "memory",
                "tool": "memory",
            },
        )

        # Memory
        workflow.add_conditional_edges(
            "memory",
            route_after_memory,
            {
                "llm": "llm",
                "rag": "rag",
                "tool": "tool",
            },
        )

        # RAG → LLM
        workflow.add_conditional_edges(
            "rag",
            route_after_rag,
            {
                "llm": "llm",
            },
        )

        # MCP → LLM
        workflow.add_conditional_edges(
            "tool",
            route_after_tool,
            {
                "llm": "llm",
            },
        )

        # LLM → Response
        workflow.add_edge(
            "llm",
            "response",
        )

        # Response → End
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