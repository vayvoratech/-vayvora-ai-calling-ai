"""Reciprocal Rank Fusion for hybrid JB Infra retrieval."""

from __future__ import annotations

from typing import Dict, List

from src.rag.config import rag_config
from src.rag.schemas import RetrievalResult


class ReciprocalRankFusion:
    """Combine dense and keyword retrieval rankings using RRF."""

    def __init__(self, k: int | None = None) -> None:
        self.k = k if k is not None else rag_config.rrf_k

    def fuse(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Fuse vector and keyword rankings."""

        combined: Dict[str, Dict] = {}

        for rank, result in enumerate(vector_results, start=1):
            item = combined.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "keyword_rank": None,
                },
            )

            item["rrf_score"] += 1.0 / (
                self.k + rank
            )

            item["vector_rank"] = rank

        for rank, result in enumerate(keyword_results, start=1):
            item = combined.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "keyword_rank": None,
                },
            )

            item["rrf_score"] += 1.0 / (
                self.k + rank
            )

            item["keyword_rank"] = rank

        fused: List[RetrievalResult] = []

        for item in combined.values():
            original = item["result"]

            metadata = {
                **original.metadata,
                "retrieval_method": "hybrid",
                "rrf_score": item["rrf_score"],
                "vector_rank": item["vector_rank"],
                "keyword_rank": item["keyword_rank"],
            }

            # Preserve dense cosine similarity when available.
            cosine_score = metadata.get(
                "cosine_score"
            )

            if cosine_score is not None:
                score = float(cosine_score)
            else:
                score = min(
                    1.0,
                    item["rrf_score"] * 25.0,
                )

            fused.append(
                RetrievalResult(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    score=score,
                    source=original.source,
                    category=original.category,
                    section=original.section,
                    metadata=metadata,
                )
            )

        fused.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return fused[:top_k]


__all__ = [
    "ReciprocalRankFusion",
]