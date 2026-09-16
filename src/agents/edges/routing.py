from src.agents.state import AgentState


def route_after_router(state: AgentState) -> str:
    """
    Route after the fast deterministic router.
    """

    route = state.get(
        "route",
        "llm_router",
    )

    if route == "direct":
        return "response"

    if route == "llm_router":
        return "llm_router"

    if route == "rag":
        return "memory"

    if route == "mcp":
        return "memory"

    return "memory"


def route_after_llm_router(state: AgentState) -> str:
    """
    Route after the LLM fallback router.
    """

    route = state.get(
        "route",
        "llm",
    )

    if route == "direct":
        return "response"

    if route == "rag":
        return "memory"

    if route == "mcp":
        return "memory"

    return "memory"


def route_after_memory(state: AgentState) -> str:
    """
    Decide what happens after conversation memory retrieval.
    """

    route = state.get(
        "route",
        "llm",
    )

    # Normal LLM conversation.
    if route == "llm":
        return "llm"

    # Memory can satisfy an information request.
    if state.get("memory_hit"):
        return "llm"

    # Company knowledge request.
    if route == "rag":
        return "rag"

    # External action/live data.
    if route == "mcp":
        return "tool"

    return "llm"


def route_after_rag(state: AgentState) -> str:
    """
    RAG context always goes to the LLM.
    """

    return "llm"


def route_after_tool(state: AgentState) -> str:
    """
    MCP result always goes to the LLM.
    """

    return "llm"