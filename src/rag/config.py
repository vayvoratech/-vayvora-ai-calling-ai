import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """
    Central configuration for the RAG system.

    All RAG-specific configuration should come from environment
    variables rather than being hardcoded throughout the codebase.
    """

    # Knowledge base
    knowledge_base_path: str = os.getenv(
        "RAG_KNOWLEDGE_BASE_PATH",
        "knowledge_base",
    )

    # Redis
    redis_url: str = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379",
    )

    redis_index_name: str = os.getenv("RAG_REDIS_INDEX", "vayvora_rag_chunks_idx")
    redis_key_prefix: str = os.getenv("RAG_REDIS_PREFIX", "vayvora:rag:chunk:")

    # Retrieval
    top_k: int = int(
        os.getenv("RAG_TOP_K", "10")
    )

    final_top_k: int = int(
        os.getenv("RAG_FINAL_TOP_K", "5")
    )

    # Minimum similarity/relevance threshold
    relevance_threshold: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "4.0"))
    

    # Chunking
    chunk_size: int = int(
        os.getenv("RAG_CHUNK_SIZE", "800")
    )

    chunk_overlap: int = int(
        os.getenv("RAG_CHUNK_OVERLAP", "120")
    )

    # Embedding
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "all-MiniLM-L6-v2",
    )

    embedding_dimension: int = int(
        os.getenv("RAG_EMBEDDING_DIMENSION", "384")
    )


rag_config = RAGConfig()