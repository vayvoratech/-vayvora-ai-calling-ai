"""Hybrid RAG retrieval service for the JB Infra demo MVP."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from src.rag.config import rag_config
from src.rag.embeddings import (
    BaseEmbeddingProvider,
    get_embedding_provider,
)
from src.rag.retrieval.fusion import ReciprocalRankFusion
from src.rag.retrieval.keyword_index import (
    KnowledgeBaseKeywordIndex,
    get_keyword_index,
)
from src.rag.retrieval.reranker import (
    Reranker,
    get_reranker,
)
from src.rag.retrieval.vector_store import RAGVectorStore
from src.rag.schemas import RAGResponse, RetrievalResult


class RAGRetriever:
    """Hybrid JB Infra retrieval pipeline.

    Pipeline:
        1. Query normalization
        2. Dense vector retrieval
        3. BM25/keyword retrieval
        4. Reciprocal Rank Fusion
        5. Optional reranking
        6. Relevance threshold gating
    """

    def __init__(
        self,
        vector_store: Optional[RAGVectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        keyword_retriever: Optional[KnowledgeBaseKeywordIndex] = None,
        reranker: Optional[Reranker] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
    ):
        self.vector_store = vector_store or RAGVectorStore()

        self.embedding_provider = (
            embedding_provider
            or get_embedding_provider()
        )

        self.reranker = reranker or get_reranker()

        self.fusion = (
            fusion
            or ReciprocalRankFusion()
        )

        self.keyword_retriever = (
            keyword_retriever
            or get_keyword_index("default")
        )

    @staticmethod
    def _expand_query(
        query: str,
        domain=None,
    ) -> str:
        """Normalize a voice query without injecting unrelated company data."""

        normalized = query.strip()

        if not normalized:
            return ""

        # The JB Infra MVP intentionally does not inject
        # Vayvora/EduSaaS/company-specific text into queries.
        return normalized

    @staticmethod
    def _belongs_to_tenant(
        result: RetrievalResult,
        tenant_id: str,
    ) -> bool:
        """Validate the retrieval result namespace."""

        if tenant_id in (
            "default",
            "all",
            "general",
        ):
            return True

        result_tenant = result.metadata.get(
            "tenant_id"
        )

        if result_tenant is None:
            return True

        return (
            str(result_tenant).lower()
            == tenant_id.lower()
        )

    @staticmethod
    def _build_context(
        results: List[RetrievalResult],
    ) -> str:
        """Build grounded context from retrieved chunks."""

        if not results:
            return ""

        context_blocks = []

        for idx, item in enumerate(
            results,
            start=1,
        ):
            source = (
                item.source
                or item.metadata.get(
                    "source_file",
                    "Knowledge Base",
                )
            )

            section = (
                item.section
                or item.metadata.get(
                    "section",
                    "General",
                )
            )

            context_blocks.append(
                f"[Document Source {idx} - "
                f"{source} | Section: {section}]\n"
                f"{item.text.strip()}"
            )

        return "\n\n".join(
            context_blocks
        )

    async def search(
        self,
        query: str,
        tenant_id: str = "default",
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """Run hybrid retrieval for a JB Infra query."""

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if not tenant_id or not tenant_id.strip():
            raise ValueError(
                "tenant_id is required."
            )

        k = (
            top_k
            if top_k is not None
            else rag_config.top_k
        )

        cutoff = (
            relevance_threshold
            if relevance_threshold is not None
            else rag_config.relevance_threshold
        )

        enriched_query = self._expand_query(
            query
        )

        # --------------------------------------------------------------
        # 1. Dense vector embedding
        # --------------------------------------------------------------

        query_vector = (
            self.embedding_provider.embed_text(
                enriched_query
            )
        )

        # --------------------------------------------------------------
        # 2. Dense vector search
        # --------------------------------------------------------------

        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
            tenant_id=tenant_id,
        )

        vector_results = [
            result
            for result in vector_results
            if self._belongs_to_tenant(
                result,
                tenant_id,
            )
        ]

        # --------------------------------------------------------------
        # 3. Keyword/BM25 search
        # --------------------------------------------------------------

        if tenant_id in (
            "default",
            "all",
            "general",
        ):
            keyword_retriever = (
                self.keyword_retriever
            )
        else:
            keyword_retriever = (
                get_keyword_index(
                    tenant_id
                )
            )

        keyword_results = (
            keyword_retriever.search(
                query=enriched_query,
                top_k=k,
            )
        )

        keyword_results = [
            result
            for result in keyword_results
            if self._belongs_to_tenant(
                result,
                tenant_id,
            )
        ]

        # --------------------------------------------------------------
        # 4. Reciprocal Rank Fusion
        # --------------------------------------------------------------

        if (
            vector_results
            and keyword_results
        ):
            candidates = self.fusion.fuse(
                vector_results=vector_results,
                keyword_results=keyword_results,
                top_k=k,
            )

        elif vector_results:
            candidates = vector_results

        else:
            candidates = keyword_results

        # --------------------------------------------------------------
        # 5. Optional reranking
        # --------------------------------------------------------------

        if self.reranker and candidates:
            final_candidates = (
                self.reranker.rerank(
                    query=enriched_query,
                    results=candidates,
                    top_k=rag_config.final_top_k,
                )
            )
        else:
            final_candidates = candidates[
                : rag_config.final_top_k
            ]

        # --------------------------------------------------------------
        # 6. Relevance gate
        # --------------------------------------------------------------

        relevant_results = [
            result
            for result in final_candidates
            if result.score >= cutoff
        ]

        context = self._build_context(
            relevant_results
        )

        return RAGResponse(
            query=query,
            results=relevant_results,
            context=context,
            has_relevant_context=bool(
                relevant_results
            ),
            metadata={
                "tenant_id": tenant_id,
                "enriched_query": enriched_query,
                "vector_results": len(
                    vector_results
                ),
                "keyword_results": len(
                    keyword_results
                ),
                "total_candidates": len(
                    candidates
                ),
                "relevant_chunks": len(
                    relevant_results
                ),
                "top_k": k,
                "final_top_k": (
                    rag_config.final_top_k
                ),
                "relevance_threshold": cutoff,
                "reranked": bool(
                    self.reranker
                ),
            },
        )


@lru_cache(maxsize=1)
def get_rag_retriever() -> RAGRetriever:
    """Return the cached JB Infra RAG retriever."""

    return RAGRetriever()


__all__ = [
    "RAGRetriever",
    "get_rag_retriever",
]