"""Grounded Hybrid RAG package supporting dense vector search, BM25, RRF, reranking, and ingestion."""

from src.rag.config import rag_config
from src.rag.ingestion.chunker import DocumentChunker, MarkdownChunker
from src.rag.ingestion.incremental import IncrementalIngestionService
from src.rag.ingestion.loader import MarkdownDocumentLoader
from src.rag.ingestion.manifest import DocumentManifest, ManifestStore
from src.rag.ingestion.pipeline import IngestionPipeline, KnowledgeIngestionPipeline
from src.rag.models import DocumentChunk
from src.rag.schemas import RAGResponse, RetrievalResult
from src.rag.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    FastEmbedProvider,
    MockEmbeddingProvider,
    get_embedding_provider,
)
from src.rag.redis_client import RedisVectorStore
from src.rag.retrieval.fusion import ReciprocalRankFusion
from src.rag.retrieval.hybrid_search import BM25Retriever, KeywordDocument
from src.rag.retrieval.keyword_index import (
    KnowledgeBaseKeywordIndex,
    get_keyword_index,
)
from src.rag.retrieval.reranker import Reranker, get_reranker
from src.rag.retrieval.vector_store import RAGVectorStore
from src.rag.retriever import (
    GroundedContextResult,
    GroundedKnowledgeProvider,
)
from src.rag.services.fast_path import (
    FAST_PATH_RULES,
    get_deterministic_fast_path,
)
from src.rag.services.recommendation import (
    CourseRecommendationService,
    get_course_recommendations,
)
from src.rag.services.retriever import RAGRetriever, get_rag_retriever

__all__ = [
    "rag_config",
    "DocumentChunk",
    "DocumentChunker",
    "MarkdownChunker",
    "MarkdownDocumentLoader",
    "DocumentManifest",
    "ManifestStore",
    "IncrementalIngestionService",
    "IngestionPipeline",
    "KnowledgeIngestionPipeline",
    "BaseEmbeddingProvider",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "MockEmbeddingProvider",
    "get_embedding_provider",
    "RedisVectorStore",
    "RAGVectorStore",
    "BM25Retriever",
    "KeywordDocument",
    "KnowledgeBaseKeywordIndex",
    "get_keyword_index",
    "ReciprocalRankFusion",
    "Reranker",
    "get_reranker",
    "GroundedContextResult",
    "GroundedKnowledgeProvider",
    "RetrievalResult",
    "RAGResponse",
    "RAGRetriever",
    "get_rag_retriever",
    "FAST_PATH_RULES",
    "get_deterministic_fast_path",
    "CourseRecommendationService",
    "get_course_recommendations",
]
