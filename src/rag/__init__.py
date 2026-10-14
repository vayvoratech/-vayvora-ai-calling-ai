"""JB Infra RAG package.

Provides document ingestion, embeddings, retrieval, vector storage,
and grounded context generation for the JB Infra AI calling demo MVP.
"""

from src.rag.config import RAGConfig, rag_config
from src.rag.models import DocumentChunk
from src.rag.redis_client import RedisVectorStore
from src.rag.retriever import (
    GroundedContextResult,
    GroundedKnowledgeProvider,
    RAGRetriever,
)
from src.rag.schemas import (
    RAGResponse,
    RetrievalResult,
)

__all__ = [
    "RAGConfig",
    "rag_config",
    "DocumentChunk",
    "RedisVectorStore",
    "GroundedContextResult",
    "GroundedKnowledgeProvider",
    "RAGRetriever",
    "RetrievalResult",
    "RAGResponse",
]