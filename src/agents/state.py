"""
State definitions for Vayvora AI Voice Calling Agent.
"""

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    # ─────────────────────────────────────────────
    # CALL / SESSION METADATA
    # ─────────────────────────────────────────────
    call_id: str
    session_id: str

    # ─────────────────────────────────────────────
    # USER VOICE INPUT
    # ─────────────────────────────────────────────
    user_input: str

    # ─────────────────────────────────────────────
    # CONVERSATION HISTORY
    # ─────────────────────────────────────────────
    messages: List[Dict[str, Any]]

    # ─────────────────────────────────────────────
    # DENSE VECTOR SEMANTIC ROUTER
    # ─────────────────────────────────────────────
    # Possible routes:
    #   'direct'  -> Immediate conversational greeting/courtesy
    #   'rag'     -> Organizational knowledge base retrieval
    #   'mcp'     -> External action or live data via MCP tools
    #   'llm'     -> General conversational dialogue or reasoning
    route: str
    route_confidence: float
    similarity_scores: Dict[str, float]
    route_margin: float
    route_rationale: str
    kb_resonance: float
    routing_latency_ms: float

    # ─────────────────────────────────────────────
    # CONVERSATION MEMORY (REDIS)
    # ─────────────────────────────────────────────
    memory_context: List[str]
    memory_hit: bool

    # ─────────────────────────────────────────────
    # GROUNDED RAG KNOWLEDGE BASE
    # ─────────────────────────────────────────────
    tenant_id: str
    rag_required: bool
    rag_context: str
    rag_response: Any

    # ─────────────────────────────────────────────
    # EXTERNAL MCP TOOLS
    # ─────────────────────────────────────────────
    tool_required: bool
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: Any
    tool_confidence: float

    # ─────────────────────────────────────────────
    # SPEECH SYNTHESIS LLM EXECUTION
    # ─────────────────────────────────────────────
    llm_required: bool

    # ─────────────────────────────────────────────
    # FINAL VOICE-READY OUTPUT
    # ─────────────────────────────────────────────
    response: str
    is_complete: bool

    # ─────────────────────────────────────────────
    # ERROR & TELEMETRY
    # ─────────────────────────────────────────────
    error: str | None
