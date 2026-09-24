import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """
    Central configuration for the Vayvora RAG system.
    """

    # Knowledge base storage
    knowledge_base_path: str = os.getenv(
        "RAG_KNOWLEDGE_BASE_PATH",
        "knowledge_base",
    )

    # Redis vector store
    redis_url: str = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379",
    )
    redis_index_name: str = os.getenv("RAG_REDIS_INDEX", "vayvora_rag_chunks_idx")
    redis_key_prefix: str = os.getenv("RAG_REDIS_PREFIX", "vayvora:rag:chunk:")

    # Retrieval hyperparameters
    top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    final_top_k: int = int(os.getenv("RAG_FINAL_TOP_K", "3"))

    # Semantic relevance threshold for hybrid / cosine scores
    relevance_threshold: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.30"))

    # Low-latency voice reranking toggle
    enable_cross_encoder: bool = os.getenv("RAG_ENABLE_CROSS_ENCODER", "false").lower() == "true"

    # Chunking
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    # Embeddings
    embedding_model: str = "text-embedding-004"
    embedding_dimension: int = 768


rag_config = RAGConfig()
