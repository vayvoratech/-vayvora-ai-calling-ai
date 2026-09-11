from __future__ import annotations

from src.rag.schemas import RetrievalResult


class ReciprocalRankFusion:
    """
    Combines vector and keyword retrieval rankings.

    RRF score:

        1 / (k + rank)

    This prevents one retrieval method from dominating
    purely because its raw score has a different scale.
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

        for rank, result in enumerate(
            vector_results,
            start=1,
        ):
            fused.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "score": 0.0,
                },
            )

            fused[result.chunk_id]["score"] += (
                1.0 / (self.k + rank)
            )

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):
            fused.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "score": 0.0,
                },
            )

            fused[result.chunk_id]["score"] += (
                1.0 / (self.k + rank)
            )

        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        results = []

        for item in ranked[:top_k]:
            result = item["result"]

            results.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=item["score"],
                    source=result.source,
                    category=result.category,
                    section=result.section,
                    metadata={
                        **result.metadata,
                        "retrieval_method": "hybrid",
                    },
                )
            )

        return results