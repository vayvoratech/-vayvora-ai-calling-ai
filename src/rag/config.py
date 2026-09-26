"""RAG configuration and tunable hyperparameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RAGConfig:
    """Tunable hyperparameters and configuration for RAG retrieval and ingestion."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    relevance_threshold: float = 0.30
    top_k: int = 5
    final_top_k: int = 3
    rrf_k: int = 60
    enable_cross_encoder: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    redis_edusaas_index: str = field(
        default_factory=lambda: os.getenv("REDIS_EDUSAAS_INDEX", "idx:edusaas_vdb")
    )
    redis_vayvora_index: str = field(
        default_factory=lambda: os.getenv("REDIS_VAYVORA_INDEX", "idx:vayvora_vdb")
    )
    redis_index_name: str = "voice_rag_chunks_idx"
    redis_key_prefix: str = "voice:rag:chunk:"

    knowledge_base_dir: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parent.parent.parent / "knowledge_base")
    )
    manifest_path: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent.parent.parent / "knowledge_base" / ".manifest.json"
        )
    )


rag_config = RAGConfig()
