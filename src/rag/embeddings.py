"""Embedding providers for vector generation and similarity scoring.

Supports local FastEmbed ONNX / SentenceTransformer models (all-MiniLM-L6-v2, 384-dim)
with LRU caching, L2 normalization, batching, and deterministic mock embeddings for offline tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections import OrderedDict
import hashlib
import math
import threading
from typing import Any, Dict, List, Optional, Tuple
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

    def is_warmed_up(self) -> bool:
        """Return whether the embedding model is already warmed up."""
        return True

    def warmup(self, sample_text: str = "warmup") -> bool:
        """Warm up embedding model synchronously. Default no-op for mock/base classes."""
        return True

    async def async_warmup(self, sample_text: str = "warmup") -> bool:
        """Warm up embedding model asynchronously in worker thread."""
        return await asyncio.to_thread(self.warmup, sample_text)


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

    _MODEL_CACHE: Dict[str, Any] = {}
    _model_lock: threading.Lock = threading.Lock()

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
        self._is_warmed_up: bool = False
        self._warmup_lock: threading.Lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return
            if self.model_name in self._MODEL_CACHE:
                self._model = self._MODEL_CACHE[self.model_name]
                return

            # Attempt sentence_transformers first if installed
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Initializing SentenceTransformer model: %s", self.model_name)
                loaded_model = SentenceTransformer(self.model_name)
                self._MODEL_CACHE[self.model_name] = loaded_model
                self._model = loaded_model
                return
            except ImportError:
                pass

            # Use FastEmbed ONNX runtime
            try:
                from fastembed import TextEmbedding
                logger.info("Initializing FastEmbed embedding model: %s", self.model_name)
                loaded_model = TextEmbedding(model_name=self.model_name)
                self._MODEL_CACHE[self.model_name] = loaded_model
                self._model = loaded_model
            except Exception as exc:
                logger.error("Failed to load embedding model '%s': %s", self.model_name, exc)
                raise RuntimeError(f"Could not initialize embedding provider: {exc}") from exc

    def is_warmed_up(self) -> bool:
        """Check if model has been warmed up."""
        return self._is_warmed_up

    def warmup(self, sample_text: str = "warmup") -> bool:
        """Warm up FastEmbed ONNX model synchronously using harmless fixed input.

        Executes model loading and an initial inference pass so that ONNX execution
        graph compilation and runtime memory allocations occur before user queries.
        Thread-safe, idempotent, and privacy-safe.
        """
        with self._warmup_lock:
            if self._is_warmed_up:
                return True
            try:
                self._ensure_model()
                # Perform single inference pass using harmless fixed string
                _ = self.embed_text(sample_text)
                self._is_warmed_up = True
                logger.info("FastEmbed ONNX embedding model warmup complete for '%s'", self.model_name)
                return True
            except Exception as exc:
                # Log privacy-safe error without any caller utterances
                logger.error("FastEmbed ONNX embedding model warmup failed for '%s': %s", self.model_name, exc)
                self._is_warmed_up = False
                return False

    async def async_warmup(self, sample_text: str = "warmup") -> bool:
        """Asynchronously warm up embedding model in a background worker thread.

        Does not block the asyncio event loop.
        """
        return await asyncio.to_thread(self.warmup, sample_text)

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


_embedding_provider_instances: Dict[Tuple[str, int, bool], BaseEmbeddingProvider] = {}
_provider_lock: threading.Lock = threading.Lock()


def get_embedding_provider(
    settings: Optional[Settings] = None, force_mock: bool = False
) -> BaseEmbeddingProvider:
    """Factory creating configured embedding provider with LRU cache and singleton reuse."""
    cfg = settings or get_settings()
    if force_mock:
        return MockEmbeddingProvider(dimension=cfg.rag_embedding_dimension)

    key = (cfg.rag_embedding_model, cfg.rag_embedding_dimension, force_mock)
    with _provider_lock:
        if key in _embedding_provider_instances:
            return _embedding_provider_instances[key]

        try:
            provider = EmbeddingProvider(
                model_name=cfg.rag_embedding_model,
                dimension=cfg.rag_embedding_dimension,
            )
        except Exception as exc:
            logger.warning("Falling back to MockEmbeddingProvider due to: %s", exc)
            provider = MockEmbeddingProvider(dimension=cfg.rag_embedding_dimension)
        _embedding_provider_instances[key] = provider
        return provider


def clear_embedding_provider_cache() -> None:
    """Clear cached embedding providers and models for test isolation."""
    with _provider_lock:
        _embedding_provider_instances.clear()
    with EmbeddingProvider._model_lock:
        EmbeddingProvider._MODEL_CACHE.clear()
