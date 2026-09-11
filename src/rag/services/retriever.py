from __future__ import annotations

from functools import lru_cache

from src.rag.config import rag_config
from src.rag.embeddings.provider import get_embedding_provider
from src.rag.retrieval.fusion import ReciprocalRankFusion
from src.rag.retrieval.keyword_index import get_keyword_index
from src.rag.retrieval.reranker import get_reranker
from src.rag.retrieval.vector_store import RAGVectorStore
from src.rag.schemas import RAGResponse, RetrievalResult


class RAGRetriever:
    def __init__(
        self,
        vector_store: RAGVectorStore | None = None,
        reranker=None,
        fusion=None,
    ):
        self.vector_store = vector_store or RAGVectorStore()
        self.embedding_provider = get_embedding_provider()
        self.reranker = reranker or get_reranker()
        self.fusion = fusion or ReciprocalRankFusion()

        # Current system uses one local knowledge base.
        # Multi-tenant keyword indexes can be added later.
        self.keyword_retriever = get_keyword_index("default")

    @staticmethod
    def _expand_query(query: str) -> str:
        # Normalize common punctuation so that:
        # "Tell me about Vayvora"
        # "Tell me about Vayvora."
        # "Tell me about Vayvora?"
        # all produce the same retrieval expansion.
        normalized = query.lower().strip().rstrip("?!.,;:")

        company_queries = {
            "what is vayvora",
            "tell me about vayvora",
            "who is vayvora",
            "what is vayvora technology",
            "tell me about vayvora technology",
            "about vayvora",
            "about vayvora technology",
        }

        if normalized in company_queries:
            return (
                f"{query} Vayvora Technology company overview "
                "about company company name company approach"
            )

        return query

    @staticmethod
    def _belongs_to_tenant(
        result: RetrievalResult,
        tenant_id: str,
    ) -> bool:
        result_tenant = result.metadata.get("tenant_id")

        if result_tenant is None:
            return True

        return result_tenant == tenant_id

    @staticmethod
    def _build_context(
        results: list[RetrievalResult],
    ) -> str:
        if not results:
            return ""

        context_parts = []

        for index, result in enumerate(results, start=1):
            context_parts.append(
                f"[Source {index}]\n"
                f"File: {result.source}\n"
                f"Section: {result.section or 'N/A'}\n"
                f"Content:\n{result.text}"
            )

        return "\n\n".join(context_parts)

    async def search(
        self,
        query: str,
        tenant_id: str = "default",
        top_k: int | None = None,
    ) -> RAGResponse:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required.")

        retrieval_k = top_k or rag_config.top_k

        if retrieval_k < 1:
            raise ValueError("top_k must be greater than zero.")

        retrieval_query = self._expand_query(query)

        # ---------------------------------------------------------
        # 1. Generate query embedding
        # ---------------------------------------------------------

        query_vector = self.embedding_provider.embed_text(
            retrieval_query
        )

        # ---------------------------------------------------------
        # 2. Vector search
        # ---------------------------------------------------------

        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=retrieval_k,
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

        # ---------------------------------------------------------
        # 3. Keyword / BM25 search
        # ---------------------------------------------------------

        keyword_results = self.keyword_retriever.search(
            query=retrieval_query,
            top_k=retrieval_k,
        )

        keyword_results = [
            result
            for result in keyword_results
            if self._belongs_to_tenant(
                result,
                tenant_id,
            )
        ]

        # ---------------------------------------------------------
        # 4. Hybrid fusion
        # ---------------------------------------------------------

        if vector_results and keyword_results:
            candidates = self.fusion.fuse(
                vector_results=vector_results,
                keyword_results=keyword_results,
                top_k=retrieval_k,
            )

        elif vector_results:
            candidates = vector_results

        else:
            candidates = keyword_results

        # ---------------------------------------------------------
        # 5. Cross-encoder reranking
        #
        # Use the expanded retrieval query here as well.
        # This keeps retrieval and reranking aligned.
        # ---------------------------------------------------------

        if self.reranker and candidates:
            final_results = self.reranker.rerank(
                query=retrieval_query,
                results=candidates,
                top_k=rag_config.final_top_k,
            )

        else:
            final_results = candidates[:rag_config.final_top_k]

        # ---------------------------------------------------------
        # 6. Relevance gate
        #
        # Cross-encoder scores are logits, not probabilities.
        # The current calibrated baseline is 4.0.
        # ---------------------------------------------------------

        relevance_threshold = rag_config.relevance_threshold

        relevant_results = [
            result
            for result in final_results
            if result.score >= relevance_threshold
        ]

        # ---------------------------------------------------------
        # 7. Build grounded context
        # ---------------------------------------------------------

        context = self._build_context(
            relevant_results
        )

        # ---------------------------------------------------------
        # 8. Return structured RAG response
        # ---------------------------------------------------------

        return RAGResponse(
            query=query,
            results=relevant_results,
            context=context,
            has_relevant_context=bool(
                relevant_results
            ),
            metadata={
                "tenant_id": tenant_id,
                "retrieval_query": retrieval_query,
                "vector_results": len(vector_results),
                "keyword_results": len(keyword_results),
                "candidate_results": len(candidates),
                "final_results": len(final_results),
                "relevant_results": len(relevant_results),
                "top_k": retrieval_k,
                "final_top_k": rag_config.final_top_k,
                "relevance_threshold": relevance_threshold,
                "reranked": bool(self.reranker),
            },
        )


@lru_cache(maxsize=1)
def get_rag_retriever() -> RAGRetriever:
    return RAGRetriever()