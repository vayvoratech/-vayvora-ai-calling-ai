from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.config import rag_config


class EmbeddingProvider:
    """
    Centralized dense vector embedding provider for Vayvora RAG and semantic routing.
    Loads model once and caches unit-normalized vectors for ultra-low latency voice responses.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or rag_config.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_embedding_dimension()
        self._cache: dict[str, list[float]] = {}

        if self.dimension != rag_config.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: model={self.dimension}, "
                f"configured={rag_config.embedding_dimension}"
            )

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        cleaned = text.strip()
        if cleaned in self._cache:
            return self._cache[cleaned]

        vector = self.model.encode(
            cleaned,
            normalize_embeddings=True,
        )
        result = vector.astype(np.float32).tolist()
        if len(self._cache) < 4096:
            self._cache[cleaned] = result
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Document list contains empty text.")

        results: list[list[float]] = []
        to_encode: list[str] = []
        indices: list[int] = []

        for idx, text in enumerate(texts):
            cleaned = text.strip()
            if cleaned in self._cache:
                results.append(self._cache[cleaned])
            else:
                results.append([])
                to_encode.append(cleaned)
                indices.append(idx)

        if to_encode:
            vectors = self.model.encode(
                to_encode,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
            for idx, vec in zip(indices, vectors):
                arr = vec.astype(np.float32).tolist()
                results[idx] = arr
                if len(self._cache) < 4096:
                    self._cache[to_encode[indices.index(idx)]] = arr

        return results


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider()
