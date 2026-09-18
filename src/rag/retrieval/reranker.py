from __future__ import annotations

from functools import lru_cache
import os
from typing import List

from src.rag.config import rag_config
from src.rag.schemas import RetrievalResult


class Reranker:
    """
    Ultra-low latency reranker for Vayvora AI Voice Calling.
    Provides fast Bi-Encoder Cosine + RRF Hybrid Scoring (<0.1ms) by default,
    with an optional lazy-loaded Cross-Encoder mode when explicitly configured.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self._model = None
        self._prediction_cache: dict[str, float] = {}

    def _get_cross_encoder(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = False
        return self._model if self._model is not False else None

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        if not query or not query.strip() or not results:
            return []

        top_k = max(1, top_k)

        # 1. Ultra-fast hybrid semantic scoring for real-time voice (<0.1ms)
        if not rag_config.enable_cross_encoder:
            reranked = []
            for result in results:
                cosine_sim = result.metadata.get("cosine_score")
                if cosine_sim is not None:
                    # Calibrate dense similarity score
                    final_score = float(cosine_sim)
                else:
                    # RRF score normalized to ~0.40 - 0.95
                    final_score = min(1.0, max(0.35, result.score * 25.0))

                metadata = dict(result.metadata)
                metadata["retrieval_method"] = "fast_hybrid_scored"

                reranked.append(
                    RetrievalResult(
                        chunk_id=result.chunk_id,
                        text=result.text,
                        score=final_score,
                        source=result.source,
                        category=result.category,
                        section=result.section,
                        metadata=metadata,
                    )
                )

            reranked.sort(key=lambda r: r.score, reverse=True)
            return reranked[:top_k]

        # 2. Optimized Cross-Encoder evaluation for top candidates only
        model = self._get_cross_encoder()
        if model is None:
            return results[:top_k]

        eval_candidates = results[:min(len(results), 3)]
        pairs = [(query.strip(), res.text[:256]) for res in eval_candidates]

        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception:
            return results[:top_k]

        reranked = []
        for result, score in zip(eval_candidates, scores):
            metadata = dict(result.metadata)
            metadata["retrieval_method"] = "hybrid_reranked"
            metadata["reranker_model"] = self.model_name

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
