from __future__ import annotations

from functools import lru_cache
from sentence_transformers import CrossEncoder

from src.rag.schemas import RetrievalResult


class Reranker:
    """
    Cross-encoder transformer reranker for RAG retrieval candidates.
    Ensures optimal semantic alignment between voice query and candidate chunks.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query or not query.strip() or not results:
            return []

        top_k = max(1, top_k)
        pairs = [(query.strip(), result.text) for result in results]

        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked: list[RetrievalResult] = []

        for result, score in zip(results, scores):
            metadata = dict(result.metadata)
            metadata.update(
                {
                    "retrieval_method": "hybrid_reranked",
                    "reranker_model": self.model_name,
                }
            )

            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=float(score),
                    source=result.source,
                    category=result.category,
                    section=result.section,
                    metadata=metadata,
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
