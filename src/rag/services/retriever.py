"""Production Hybrid RAG Retrieval Service.

Executes multi-stage hybrid retrieval:
1. Query expansion & semantic domain cue enrichment
2. Dense vector search with normalized embeddings
3. Lexical keyword retrieval via BM25
4. Hybrid rank fusion via Reciprocal Rank Fusion (RRF)
5. Semantic scoring & reranking
6. Relevance score threshold gating to eliminate hallucinations
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from src.rag.config import rag_config
from src.rag.embeddings import BaseEmbeddingProvider, get_embedding_provider
from src.rag.retrieval.fusion import ReciprocalRankFusion
from src.rag.retrieval.keyword_index import KnowledgeBaseKeywordIndex, get_keyword_index
from src.rag.retrieval.reranker import Reranker, get_reranker
from src.rag.retrieval.vector_store import RAGVectorStore
from src.rag.schemas import RAGResponse, RetrievalResult


class RAGRetriever:
    """
    Multi-stage Hybrid RAG Pipeline:
    1. First stage: Dense Vector Search + Lexical BM25 Search
    2. Fusion: Reciprocal Rank Fusion (RRF)
    3. Second stage: Ultra-low latency Reranker
    4. Relevance Gate: Score thresholding to prevent hallucinations
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
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.reranker = reranker or get_reranker()
        self.fusion = fusion or ReciprocalRankFusion()
        self.keyword_retriever = keyword_retriever or get_keyword_index("default")

    @staticmethod
    def _expand_query(query: str, domain: Optional[Any] = None) -> str:
        """Semantically enrich voice queries with domain context."""
        normalized = query.lower().strip().rstrip("?!.,;:")

        domain_cues = (
            "services", "capabilities", "pricing", "price", "cost", "rates", "rate",
            "fee", "fees", "course", "courses", "training", "bootcamp",
            "refund", "sla", "uptime", "contact", "headquarters",
            "office", "cloud", "aws", "mobile", "ai", "genai",
            "product", "products", "solution", "solutions",
            "voice", "agent", "agents", "bot", "bots",
            "intelligence", "artificial intelligence",
            "apply", "applying", "career", "careers", "job", "jobs", "role", "roles",
            "hiring", "hire", "internship", "intern", "resume", "cv", "vacancy", "vacancies",
            "assessment", "assessments", "interview", "interviews", "exam", "exams",
        )
        dom_val = domain.value.lower() if hasattr(domain, "value") else (str(domain).lower() if domain else "")
        if dom_val == "edusaas":
            if any(cue in normalized for cue in domain_cues) and "edusaas" not in normalized:
                return f"{query} EduSaaS engineering technical courses curriculum"
        else:
            if any(cue in normalized for cue in domain_cues) and "vayvora" not in normalized and "edusaas" not in normalized:
                return f"{query} Vayvora Technology enterprise software AI consultancy"

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
        if tenant_id in ("default", "all"):
            return True
        result_tenant = result.metadata.get("tenant_id")
        if result_tenant is None:
            return True
        return result_tenant.lower() == tenant_id.lower()

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
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
    ) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required.")

        k = top_k or rag_config.top_k
        cutoff = relevance_threshold if relevance_threshold is not None else rag_config.relevance_threshold
        enriched_query = self._expand_query(query)

        # 1. Dense vector embedding
        query_vector = self.embedding_provider.embed_text(enriched_query)

        # 2. Dense vector search
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
        kw_retriever = (
            get_keyword_index(tenant_id)
            if tenant_id not in ("default", "all")
            else self.keyword_retriever
        )
        keyword_results = kw_retriever.search(
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

        # 5. Reranking
        if self.reranker and candidates:
            final_candidates = self.reranker.rerank(
                query=enriched_query,
                results=candidates,
                top_k=rag_config.final_top_k,
            )
        else:
            final_candidates = candidates[:rag_config.final_top_k]

        # 6. Semantic Relevance Gate
        relevant_results = [
            res for res in final_candidates
            if res.score >= cutoff
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
                "relevance_threshold": cutoff,
                "reranked": bool(self.reranker),
            },
        )


@lru_cache(maxsize=1)
def get_rag_retriever() -> RAGRetriever:
    return RAGRetriever()
