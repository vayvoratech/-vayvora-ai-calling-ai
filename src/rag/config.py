"""RAG configuration and tunable hyperparameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    """Configuration for JB Infra RAG retrieval and ingestion."""

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Retrieval
    relevance_threshold: float = 0.60
    top_k: int = 5
    final_top_k: int = 3
    rrf_k: int = 60

    # Optional cross-encoder reranking
    enable_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Redis
    redis_url: str = field(
        default_factory=lambda: os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0",
        )
    )

    # Single RAG index for the JB Infra MVP
    redis_index_name: str = field(
        default_factory=lambda: os.getenv(
            "REDIS_RAG_INDEX",
            "idx:general_vdb",
        )
    )

    # Redis document namespace
    redis_key_prefix: str = field(
        default_factory=lambda: os.getenv(
            "REDIS_RAG_KEY_PREFIX",
            "rag:general:doc:",
        )
    )

    # Knowledge base
    knowledge_base_dir: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent.parent.parent / "knowledge_base"
        )
    )

    # Manifest
    manifest_path: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent.parent.parent
            / "knowledge_base"
            / ".manifest.json"
        )
    )


rag_config = RAGConfig()
