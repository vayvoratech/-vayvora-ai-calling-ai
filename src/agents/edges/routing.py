from src.agents.state import AgentState


def route_after_router(state: AgentState) -> str:
    """
    Decide the next Agent node after the fast local router.
    """

    route = state.get("route", "llm")

    if route == "direct":
        return "response"

    if route == "rag":
        return "rag"

    if route == "mcp":
        return "tool"

    return "llm"


def route_after_memory(state: AgentState) -> str:
    """
    Decide whether RAG is actually necessary after checking memory.

    If the requested information is already available in memory,
    skip RAG and go directly to the LLM.

    MCP is still handled separately because it may represent a
    live external action or live external data request.
    """

    route = state.get("route", "llm")

    # Normal conversation.
    if route == "llm":
        return "llm"

    # Memory already contains relevant information.
    if state.get("memory_hit"):
        return "llm"

    # RAG was requested and memory did not satisfy the request.
    if route == "rag":
        return "rag"

    # MCP request.
    if route == "mcp":
        return "tool"

    return "llm"


def route_after_rag(state: AgentState) -> str:
    """
    RAG always feeds the retrieved context into the single LLM.
    """

    return "llm"


def route_after_tool(state: AgentState) -> str:
    """
    MCP/tool results are passed into the single LLM.
    """

    return "llm"