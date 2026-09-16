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
    # fast_router
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