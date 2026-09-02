from typing import Any

from src.agents.graph import AgentGraph
from src.agents.tools.mcp_bootstrap import (
    create_agent_tool_registry,
)
from src.agents.tools.mcp_client import MCPClient


class AgentRuntime:
    """
    Long-lived runtime for Vayvora AI.

    Initializes shared infrastructure once and reuses it
    across voice calls.

    MCP connection is persistent instead of reconnecting
    for every user request.
    """

    def __init__(
        self,
        llm: Any,
        memory_store: Any = None,
        retriever: Any = None,
    ) -> None:
        self.llm = llm
        self.memory_store = memory_store
        self.retriever = retriever

        self.agent: AgentGraph | None = None
        self.mcp_client: MCPClient | None = None

    async def start(self) -> None:
        """
        Initialize MCP and the Agent.
        """

        if self.agent is not None:
            return

        tool_registry, mcp_client = (
            await create_agent_tool_registry()
        )

        self.mcp_client = mcp_client

        self.agent = AgentGraph(
            llm=self.llm,
            tool_registry=tool_registry,
            memory_store=self.memory_store,
            retriever=self.retriever,
        )

    async def run(
        self,
        state: dict,
    ) -> dict:
        """
        Execute one Agent turn.
        """

        if self.agent is None:
            await self.start()

        return await self.agent.run(state)

    async def close(self) -> None:
        """
        Gracefully shut down shared resources.
        """

        if self.mcp_client is not None:
            await self.mcp_client.close()

        self.mcp_client = None
        self.agent = None