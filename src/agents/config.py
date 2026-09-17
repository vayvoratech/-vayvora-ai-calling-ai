"""
Configuration for Vayvora AI Voice Calling Agent.

Includes hyperparameters for dense vector semantic routing, knowledge-base
semantic resonance, and semantic MCP tool selection.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    # ---------------------------------------------------------
    # Speech Synthesis LLM Parameters
    # ---------------------------------------------------------
    model_name: str = "gemini-3.5-flash-lite"
    temperature: float = 0.2
    max_tokens: int = 400

    # ---------------------------------------------------------
    # Semantic Intent Routing (Dense Vector Embeddings)
    # ---------------------------------------------------------
    enable_semantic_routing: bool = True
    
    # Softmax temperature for probability calibration across candidate routes
    router_temperature: float = 0.08
    
    # Minimum probability threshold required to trigger specialized routes
    router_confidence_threshold: float = 0.45
    
    # Minimum margin (top_score - second_score) required to avoid ambiguity fallback
    router_min_margin: float = 0.06
    
    # Balance between top-k exemplar cosine similarity and cluster centroid similarity
    router_top_k_weight: float = 0.70
    router_centroid_weight: float = 0.30

    # Weight of knowledge-base semantic resonance integrated into the RAG route score
    kb_resonance_weight: float = 0.30

    # ---------------------------------------------------------
    # Semantic Tool Selection (MCP Actions)
    # ---------------------------------------------------------
    enable_semantic_tool_selection: bool = True
    tool_similarity_threshold: float = 0.45

    # ---------------------------------------------------------
    # Execution Toggles
    # ---------------------------------------------------------
    enable_direct_responses: bool = True
    enable_llm: bool = True
    enable_rag: bool = True
    enable_mcp: bool = True

    # ---------------------------------------------------------
    # Conversation Memory & Grounded RAG Retrieval
    # ---------------------------------------------------------
    enable_memory: bool = True
    memory_top_k: int = 5
    rag_top_k: int = 6
    rag_relevance_threshold: float = 0.35

    # ---------------------------------------------------------
    # Voice Latency & Operational Safety
    # ---------------------------------------------------------
    max_response_length: int = 1500
    max_tool_calls: int = 3
    request_timeout_seconds: float = 8.0


DEFAULT_AGENT_CONFIG = AgentConfig()
