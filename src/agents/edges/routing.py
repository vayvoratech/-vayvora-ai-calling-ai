"""
LangGraph Conditional Routing Edges for Vayvora AI.

Routes workflow execution dynamically based on the continuous semantic
classification and conversation memory states.
"""

from src.agents.state import AgentState


def route_after_router(state: AgentState) -> str:
    """
    Decide the next Agent node after semantic vector classification.
    """
    route = state.get("route", "llm")

    # Fast-path for conversational greetings and short courtesies
    if route == "direct":
        return "response"

    # All non-direct queries pass through memory inspection first
    if route in ("rag", "mcp", "llm"):
        return "memory"

    return "memory"


def route_after_memory(state: AgentState) -> str:
    """
    Decide whether RAG or MCP execution is necessary after checking conversation memory.

    If the requested company knowledge is already satisfied in the multi-turn
    conversation memory, bypass expensive RAG retrieval and route straight to the LLM.

    MCP external actions are passed through to the tool node because they represent
    live external data queries or state mutations (e.g. sending messages, booking calls).
    """
    route = state.get("route", "llm")

    # General conversation
    if route == "llm":
        return "llm"

    # Memory already contains the needed organizational facts
    if route == "rag" and state.get("memory_hit"):
        return "llm"

    # Company knowledge requested and not present in memory
    if route == "rag":
        return "rag"

    # MCP external tool action
    if route == "mcp":
        return "tool"

    return "llm"


def route_after_rag(state: AgentState) -> str:
    """
    Retrieved knowledge base context feeds into the speech LLM for grounded synthesis.
    """
    return "llm"


def route_after_tool(state: AgentState) -> str:
    """
    External tool execution results feed into the speech LLM for conversational confirmation,
    or fast-path directly to response if already synthesized or prompting for missing parameters.
    """
    if state.get("is_complete") and state.get("response"):
        return "response"
    return "llm"
