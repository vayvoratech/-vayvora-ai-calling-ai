from typing import Any, TypedDict


class AgentState(TypedDict, total=False):

    # ─────────────────────────────────────────────
    # CALL / SESSION
    # ─────────────────────────────────────────────

    call_id: str
    session_id: str

    # ─────────────────────────────────────────────
    # USER INPUT
    # ─────────────────────────────────────────────

    user_input: str

    # ─────────────────────────────────────────────
    # CONVERSATION
    # ─────────────────────────────────────────────

    messages: list[dict[str, Any]]

    # ─────────────────────────────────────────────
    # ROUTING
    # ─────────────────────────────────────────────

    route: str

    # Confidence of the latest routing decision.
    route_confidence: float

    # Where the decision came from:
    #
    # fast_routerfrom typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ---------------------------------------------------------
    # Request / session
    # ---------------------------------------------------------
    call_id: str
    session_id: str
    tenant_id: str

    # ---------------------------------------------------------
    # User input
    # ---------------------------------------------------------
    user_input: str

    # ---------------------------------------------------------
    # Conversation
    # ---------------------------------------------------------
    messages: list[dict[str, Any]]

    # ---------------------------------------------------------
    # Routing
    #
    # Final routes:
    #   llm
    #   rag
    #   mcp
    #
    # llm_router is a node, NOT a route.
    # ---------------------------------------------------------
    route: str
    route_confidence: float
    route_source: str

    # ---------------------------------------------------------
    # LLM Router
    # ---------------------------------------------------------
    llm_router_required: bool

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------
    memory_context: str
    memory_hit: bool

    # ---------------------------------------------------------
    # RAG
    # ---------------------------------------------------------
    rag_required: bool
    rag_context: str
    rag_response: str

    # ---------------------------------------------------------
    # MCP / Tools
    # ---------------------------------------------------------
    tool_required: bool
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any

    # ---------------------------------------------------------
    # Final LLM
    # ---------------------------------------------------------
    llm_required: bool

    # ---------------------------------------------------------
    # Response
    # ---------------------------------------------------------
    response: str

    # ---------------------------------------------------------
    # Execution
    # ---------------------------------------------------------
    is_complete: bool
    error: str | None
    # llm_router
    # system
    route_source: str

    # Whether the LLM routing fallback is required.
    llm_router_required: bool

    # ─────────────────────────────────────────────
    # REDIS MEMORY / CACHE
    # ─────────────────────────────────────────────

    memory_context: list[str]
    memory_hit: bool

    # ─────────────────────────────────────────────
    # RAG
    # ─────────────────────────────────────────────

    tenant_id: str

    rag_required: bool
    rag_context: str
    rag_response: Any

    # ─────────────────────────────────────────────
    # MCP / TOOLS
    # ─────────────────────────────────────────────

    tool_required: bool
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any

    # ─────────────────────────────────────────────
    # LLM
    # ─────────────────────────────────────────────

    llm_required: bool

    # ─────────────────────────────────────────────
    # FINAL RESPONSE
    # ─────────────────────────────────────────────

    response: str

    # ─────────────────────────────────────────────
    # CONTROL
    # ─────────────────────────────────────────────

    is_complete: bool

    # ─────────────────────────────────────────────
    # ERROR HANDLING
    # ─────────────────────────────────────────────

    error: str | None