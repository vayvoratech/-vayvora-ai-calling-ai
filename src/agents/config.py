from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """
    Central configuration for the Vayvora AI Agent.

    Keep latency-sensitive settings here so they can be
    tuned without changing the Agent implementation.
    """

    # ─────────────────────────────────────────────
    # LLM
    # ─────────────────────────────────────────────

    model_name: str = "default"
    temperature: float = 0.2
    max_tokens: int = 500

    # ─────────────────────────────────────────────
    # ROUTING
    # ─────────────────────────────────────────────

    enable_direct_responses: bool = True
    enable_llm: bool = True
    enable_rag: bool = True
    enable_mcp: bool = True

    # ─────────────────────────────────────────────
    # MEMORY
    # ─────────────────────────────────────────────

    enable_memory: bool = True
    memory_top_k: int = 5

    # ─────────────────────────────────────────────
    # RAG
    # ─────────────────────────────────────────────

    rag_top_k: int = 5

    # ─────────────────────────────────────────────
    # RESPONSE
    # ─────────────────────────────────────────────

    max_response_length: int = 2000

    # ─────────────────────────────────────────────
    # EXECUTION SAFETY
    # ─────────────────────────────────────────────

    max_tool_calls: int = 3
    request_timeout_seconds: float = 10.0


DEFAULT_AGENT_CONFIG = AgentConfig()