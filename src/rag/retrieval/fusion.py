from __future__ import annotations

from src.rag.schemas import RetrievalResult


class ReciprocalRankFusion:
    """
    Combines dense vector and lexical BM25 retrieval rankings using RRF:
        score = 1.0 / (k + rank)
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        vector_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        fused = {}

        for rank, result in enumerate(vector_results, start=1):
            fused.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "score": 0.0,
                    "cosine_score": result.metadata.get("cosine_score", result.score),
                },
            )
            fused[result.chunk_id]["score"] += 1.0 / (self.k + rank)

        for rank, result in enumerate(keyword_results, start=1):
            fused.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "score": 0.0,
                    "cosine_score": result.metadata.get("cosine_score"),
                },
            )
            fused[result.chunk_id]["score"] += 1.0 / (self.k + rank)

        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        results = []
        for item in ranked[:top_k]:
            result = item["result"]
            meta = dict(result.metadata)
            meta["retrieval_method"] = "hybrid_rrf"
            if item.get("cosine_score") is not None:
                meta["cosine_score"] = item["cosine_score"]

            results.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=item["score"],
                    source=result.source,
                    category=result.category,
                    section=result.section,
                    metadata=meta,
                )
            )

        return results
