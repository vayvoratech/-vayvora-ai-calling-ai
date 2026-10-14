"""Reranking utilities for the JB Infra RAG demo MVP."""

from __future__ import annotations

from typing import List, Optional

from src.rag.config import rag_config
from src.rag.schemas import RetrievalResult


class Reranker:
    """Rerank hybrid retrieval results.

    Cross-encoder reranking is optional. For the demo MVP, the default
    behavior uses the retrieval score and dense cosine score when available.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
    ) -> None:
        self.model_name = (
            model_name
            or rag_config.cross_encoder_model
        )
        self._model = None

    def _load_model(self):
        """Load the cross-encoder only when explicitly enabled."""

        if self._model is not None:
            return self._model

        if not rag_config.enable_cross_encoder:
            return None

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
            )

            return self._model

        except Exception:
            # Cross-encoder is optional for the MVP.
            self._model = None
            return None

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank retrieval candidates."""

        if not results:
            return []

        model = self._load_model()

        if model is not None:
            return self._cross_encoder_rerank(
                query=query,
                results=results,
                model=model,
                top_k=top_k,
            )

        return self._fast_rerank(
            results=results,
            top_k=top_k,
        )

    @staticmethod
    def _fast_rerank(
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Fast deterministic reranking without an external model."""

        reranked: List[RetrievalResult] = []

        for result in results:
            cosine_score = result.metadata.get(
                "cosine_score"
            )

            rrf_score = result.metadata.get(
                "rrf_score"
            )

            if cosine_score is not None:
                final_score = float(
                    cosine_score
                )

            elif rrf_score is not None:
                final_score = min(
                    1.0,
                    float(rrf_score) * 25.0,
                )

            else:
                final_score = float(
                    result.score
                )

            result.score = max(
                0.0,
                min(1.0, final_score),
            )

            result.metadata = {
                **result.metadata,
                "reranking_method": "fast",
                "reranker_score": result.score,
            }

            reranked.append(result)

        reranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        if top_k is not None:
            return reranked[:top_k]

        return reranked

    @staticmethod
    def _cross_encoder_rerank(
        query: str,
        results: List[RetrievalResult],
        model,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank candidates using a sentence-transformers cross encoder."""

        pairs = [
            (
                query,
                result.text,
            )
            for result in results
        ]

        try:
            scores = model.predict(pairs)

        except Exception:
            return Reranker._fast_rerank(
                results=results,
                top_k=top_k,
            )

        reranked: List[RetrievalResult] = []

        for result, score in zip(
            results,
            scores,
        ):
            normalized_score = Reranker._normalize_score(
                float(score)
            )

            result.score = normalized_score

            result.metadata = {
                **result.metadata,
                "reranking_method": "cross_encoder",
                "reranker_score": normalized_score,
            }

            reranked.append(result)

        reranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        if top_k is not None:
            return reranked[:top_k]

        return reranked

    @staticmethod
    def _normalize_score(
        score: float,
    ) -> float:
        """Convert an arbitrary cross-encoder score into 0..1."""

        # Most cross-encoder models return logits.
        # A sigmoid provides a stable 0..1 range.
        import math

        try:
            normalized = 1.0 / (
                1.0 + math.exp(-score)
            )
        except OverflowError:
            normalized = (
                0.0
                if score < 0
                else 1.0
            )

        return max(
            0.0,
            min(1.0, normalized),
        )


def get_reranker() -> Reranker:
    """Return a configured reranker."""

    return Reranker()


__all__ = [
    "Reranker",
    "get_reranker",
]