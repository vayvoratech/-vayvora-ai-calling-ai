"""Embedding providers for vector generation and similarity scoring.

Supports local FastEmbed ONNX / SentenceTransformer models (all-MiniLM-L6-v2, 384-dim)
with LRU caching, L2 normalization, batching, and deterministic mock embeddings for offline tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import hashlib
import math
from typing import List, Optional
import numpy as np

from src.config import Settings, get_settings
from src.logging import get_logger
from src.rag.config import rag_config

logger = get_logger("rag.embeddings")


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense vector embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate a dense vector embedding for a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of texts."""
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compatibility alias for batch embedding."""
        return self.embed_batch(texts)


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, offline embedding generator producing normalized vectors."""

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_to_vector(self, text: str) -> List[float]:
        """Map text to a deterministic unit-length float vector using token feature hashing."""
        import re

        cleaned = (text or "").lower().strip()
        words = re.findall(r"\w+", cleaned)
        if not words:
            words = ["empty_placeholder"]

        vec = [0.0] * self._dim
        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 6) for x in vec]

    def embed_text(self, text: str) -> List[float]:
        return self._hash_to_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]


class EmbeddingProvider(BaseEmbeddingProvider):
    """Production embedding provider with LRU cache and L2 normalization."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        cache_size: int = 4096,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self._dim = dimension
        self.cache_size = cache_size
        self.batch_size = batch_size
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return

        # Attempt sentence_transformers first if installed
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Initializing SentenceTransformer model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            return
        except ImportError:
            pass

        # Use FastEmbed ONNX runtime
        try:
            from fastembed import TextEmbedding
            logger.info("Initializing FastEmbed embedding model: %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self.model_name, exc)
            raise RuntimeError(f"Could not initialize embedding provider: {exc}") from exc

    @property
    def dimension(self) -> int:
        return self._dim

    @staticmethod
    def _normalize(vector: List[float] | np.ndarray) -> List[float]:
        arr = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()

    def embed_text(self, text: str) -> List[float]:
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached

        self._ensure_model()

        if hasattr(self._model, "encode"):
            # SentenceTransformer
            raw = self._model.encode(text, convert_to_numpy=True)
            normalized = self._normalize(raw)
        else:
            # FastEmbed TextEmbedding
            raw = list(self._model.embed([text]))[0]
            normalized = self._normalize(raw)

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[text] = normalized
        return normalized

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                results.append(cached)
            else:
                results.append([])
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            self._ensure_model()
            if hasattr(self._model, "encode"):
                embeddings = self._model.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                for pos, emb in zip(uncached_indices, embeddings):
                    normalized = self._normalize(emb)
                    results[pos] = normalized
                    if len(self._cache) >= self.cache_size:
                        self._cache.popitem(last=False)
                    self._cache[texts[pos]] = normalized
            else:
                embeddings = list(self._model.embed(uncached_texts, batch_size=self.batch_size))
                for pos, emb in zip(uncached_indices, embeddings):
                    normalized = self._normalize(emb)
                    results[pos] = normalized
                    if len(self._cache) >= self.cache_size:
                        self._cache.popitem(last=False)
                    self._cache[texts[pos]] = normalized

        return results


# Backward compatibility alias
FastEmbedProvider = EmbeddingProvider


def get_embedding_provider(
    settings: Optional[Settings] = None, force_mock: bool = False
) -> BaseEmbeddingProvider:
    """Factory creating configured embedding provider with LRU cache."""
    cfg = settings or get_settings()
    if force_mock:
        return MockEmbeddingProvider(dimension=cfg.rag_embedding_dimension)

    try:
        return EmbeddingProvider(
            model_name=cfg.rag_embedding_model,
            dimension=cfg.rag_embedding_dimension,
        )
    except Exception as exc:
        logger.warning("Falling back to MockEmbeddingProvider due to: %s", exc)
        return MockEmbeddingProvider(dimension=cfg.rag_embedding_dimension)
