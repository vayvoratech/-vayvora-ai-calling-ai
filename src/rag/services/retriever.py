"""
Production RAG Retrieval Service with Semantic Knowledge-Base Integration.

Executes multi-stage hybrid retrieval for Vayvora AI Voice Calling:
1. Dense vector search via Redis VSet with SentenceTransformer embeddings
2. Lexical keyword retrieval via BM25
3. Hybrid rank fusion via Reciprocal Rank Fusion (RRF)
4. Semantic cross-encoder reranking
5. Relevance score threshold gating to eliminate hallucinations
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from src.rag.config import rag_config
from src.rag.embeddings.provider import get_embedding_provider
from src.rag.retrieval.fusion import ReciprocalRankFusion
from src.rag.retrieval.keyword_index import get_keyword_index
from src.rag.retrieval.reranker import get_reranker
from src.rag.retrieval.vector_store import RAGVectorStore
from src.rag.schemas import RAGResponse, RetrievalResult


class RAGRetriever:
    """
    Two-stage Hybrid RAG Pipeline with Semantic Query Enrichment:
    1. First stage: Dense Redis vector search (VSIM) + Lexical BM25 search
    2. Fusion: Reciprocal Rank Fusion (RRF)
    3. Second stage: Cross-Encoder Transformer Reranker
    4. Relevance Gate: Score thresholding to eliminate hallucinations
    """

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
        self.keyword_retriever = get_keyword_index("default")

    @staticmethod
    def _expand_query(query: str) -> str:
        """
        Semantically enrich concise or conversational voice queries
        with knowledge base domain context before vector generation.
        """
        from src.agents.nodes.router_node import RouterNode
        if RouterNode.is_general_query(query):
            return query

        normalized = query.lower().strip().rstrip("?!.,;:")

        domain_cues = (
            "services", "capabilities", "pricing", "price", "cost", "rates", "rate",
            "fee", "fees", "course", "courses", "training", "bootcamp",
            "refund", "sla", "uptime", "contact", "headquarters",
            "office", "cloud", "aws", "mobile", "ai", "genai",
            "apply", "applying", "career", "careers", "job", "jobs", "role", "roles",
            "hiring", "hire", "internship", "intern", "resume", "cv", "vacancy", "vacancies",
            "assessment", "assessments", "interview", "interviews", "exam", "exams", "examination",
        )
        if any(cue in normalized for cue in domain_cues) and "vayvora" not in normalized:
            return f"{query} Vayvora Technology enterprise software AI consultancy"

        # Specific company identity queries
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
            return f"{query} Vayvora Technology company overview about company vision services"

        return query

    @staticmethod
    def _belongs_to_tenant(result: RetrievalResult, tenant_id: str) -> bool:
        result_tenant = result.metadata.get("tenant_id")
        if result_tenant is None:
            return True
        return result_tenant == tenant_id

    @staticmethod
    def _build_context(results: List[RetrievalResult]) -> str:
        if not results:
            return ""

        context_blocks = []
        for idx, item in enumerate(results, start=1):
            context_blocks.append(
                f"[Document Source {idx} - {item.source} | Section: {item.section or 'General'}]\n"
                f"{item.text.strip()}"
            )
        return "\n\n".join(context_blocks)

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

        k = top_k or rag_config.top_k
        enriched_query = self._expand_query(query)

        # 1. Dense vector embedding
        query_vector = self.embedding_provider.embed_text(enriched_query)

        # 2. Redis vector search
        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
            tenant_id=tenant_id,
        )
        vector_results = [
            res for res in vector_results
            if self._belongs_to_tenant(res, tenant_id)
        ]

        # 3. BM25 keyword search
        keyword_results = self.keyword_retriever.search(
            query=enriched_query,
            top_k=k,
        )
        keyword_results = [
            res for res in keyword_results
            if self._belongs_to_tenant(res, tenant_id)
        ]

        # 4. Hybrid Reciprocal Rank Fusion
        if vector_results and keyword_results:
            candidates = self.fusion.fuse(
                vector_results=vector_results,
                keyword_results=keyword_results,
                top_k=k,
            )
        elif vector_results:
            candidates = vector_results
        else:
            candidates = keyword_results

        # 5. Cross-encoder transformer reranking
        if self.reranker and candidates:
            final_candidates = self.reranker.rerank(
                query=enriched_query,
                results=candidates,
                top_k=rag_config.final_top_k,
            )
        else:
            final_candidates = candidates[:rag_config.final_top_k]

        # 6. Semantic Relevance Gate
        relevance_threshold = rag_config.relevance_threshold
        relevant_results = [
            res for res in final_candidates
            if res.score >= relevance_threshold
        ]

        context = self._build_context(relevant_results)

        return RAGResponse(
            query=query,
            results=relevant_results,
            context=context,
            has_relevant_context=bool(relevant_results),
            metadata={
                "tenant_id": tenant_id,
                "enriched_query": enriched_query,
                "vector_results": len(vector_results),
                "keyword_results": len(keyword_results),
                "total_candidates": len(candidates),
                "relevant_chunks": len(relevant_results),
                "top_k": k,
                "final_top_k": rag_config.final_top_k,
                "relevance_threshold": relevance_threshold,
                "reranked": bool(self.reranker),
            },
        )


@lru_cache(maxsize=1)
def get_rag_retriever() -> RAGRetriever:
    return RAGRetriever()
