from src.agents.state import AgentState


def route_after_llm_router(state: AgentState) -> str:
    """
    Route after the LLM Router.

    The LLM Router produces exactly one final route:

        llm
        rag
        mcp
    """

    route = state.get("route", "llm")

    if route == "rag":
        return "rag"

    if route == "mcp":
        return "tool"

    return "llm"


def route_after_memory(state: AgentState) -> str:
    """
    Decide whether memory can satisfy the request.

    RAG:
        If memory does not satisfy the request, retrieve from RAG.

    MCP:
        Always execute the external tool because it may involve
        live data or an external action.

    LLM:
        Go directly to the final LLM.
    """

    route = state.get("route", "llm")

    # ---------------------------------------------------------
    # Normal LLM request
    # ---------------------------------------------------------

    if route == "llm":
        return "llm"

    # ---------------------------------------------------------
    # MCP request
    #
    # MCP may require live external data/action.
    # Do not let normal conversation memory replace it.
    # ---------------------------------------------------------

    if route == "mcp":
        return "tool"

    # ---------------------------------------------------------
    # RAG request
    #
    # If memory already contains relevant information,
    # allow the LLM to answer using memory.
    # Otherwise perform RAG retrieval.
    # ---------------------------------------------------------

    if route == "rag":
        if state.get("memory_hit"):
            return "llm"

        return "rag"

    return "llm"


def route_after_rag(state: AgentState) -> str:
    """
    RAG result always goes to the final LLM.
    """

    return "llm"


def route_after_tool(state: AgentState) -> str:
    """
    MCP/tool result always goes to the final LLM.
    """

    return "llm"