"""Reciprocal Rank Fusion (RRF) combining dense vector and lexical rankings."""

from __future__ import annotations

from typing import Dict, List, Optional
from src.rag.config import rag_config
from src.rag.schemas import RetrievalResult


class ReciprocalRankFusion:
    """
    Combines dense vector and lexical BM25 retrieval rankings using Reciprocal Rank Fusion:
        score = sum(1.0 / (k + rank))
    """

    def __init__(self, k: Optional[int] = None):
        self.k = k if k is not None else rag_config.rrf_k

    def fuse(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        fused: Dict[str, dict] = {}

        # 1. Score vector results
        for rank, result in enumerate(vector_results, start=1):
            fused.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "score": 0.0,
                    "cosine_score": result.metadata.get("cosine_score", result.score),
                    "vector_rank": rank,
                },
            )
            fused[result.chunk_id]["score"] += 1.0 / (self.k + rank)

        # 2. Score keyword/BM25 results
        for rank, result in enumerate(keyword_results, start=1):
            fused.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "score": 0.0,
                    "cosine_score": result.metadata.get("cosine_score"),
                    "keyword_rank": rank,
                },
            )
            fused[result.chunk_id]["score"] += 1.0 / (self.k + rank)
            fused[result.chunk_id]["keyword_rank"] = rank

        # 3. Sort by aggregated RRF score descending
        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        results: List[RetrievalResult] = []
        for item in ranked[:top_k]:
            result = item["result"]
            meta = dict(result.metadata)
            meta["retrieval_method"] = "hybrid_rrf"
            meta["rrf_score"] = item["score"]
            if item.get("cosine_score") is not None:
                meta["cosine_score"] = item["cosine_score"]
            if "vector_rank" in item:
                meta["vector_rank"] = item["vector_rank"]
            if "keyword_rank" in item:
                meta["keyword_rank"] = item["keyword_rank"]

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
