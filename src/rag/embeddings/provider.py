from functools import lru_cache
import numpy as np
from fastembed import TextEmbedding

from src.rag.config import rag_config


class EmbeddingProvider:
    """
    Ultra-lightweight dense vector embedding provider for Vayvora RAG and semantic routing.
    Powered by FastEmbed (ONNX Runtime).
    - 0 API Keys / 0 Network Calls / 0 Cloud Version Mismatches
    - Consumes only ~40MB RAM (Render 512MB safe)
    - Replaces PyTorch, Transformers, and Google Cloud Embeddings.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        # Loads a tiny ~60MB quantized ONNX model into memory
        self.model = TextEmbedding(model_name=self.model_name)
        # BAAI/bge-small-en-v1.5 produces 384 dimensions
        self.dimension = 384
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        """Unit L2-normalization for cosine dot products."""
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        cleaned = text.strip()
        if cleaned in self._cache:
            return self._cache[cleaned]

        # FastEmbed returns a generator of numpy arrays
        generator = self.model.embed([cleaned])
        raw_vector = list(generator)[0].tolist()
        result = self._normalize(raw_vector)

        if len(self._cache) < 4096:
            self._cache[cleaned] = result
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Document list contains empty text.")

        results: list[list[float]] = [[] for _ in texts]
        to_encode: list[str] = []
        indices: list[int] = []

        for idx, text in enumerate(texts):
            cleaned = text.strip()
            if cleaned in self._cache:
                results[idx] = self._cache[cleaned]
            else:
                to_encode.append(cleaned)
                indices.append(idx)

        if to_encode:
            # Batch embedding via FastEmbed
            generator = self.model.embed(to_encode)
            raw_vectors = [v.tolist() for v in generator]

            for idx, raw_vec in zip(indices, raw_vectors):
                normalized_vec = self._normalize(raw_vec)
                results[idx] = normalized_vec
                cached_text = texts[idx].strip()
                if len(self._cache) < 4096:
                    self._cache[cached_text] = normalized_vec

        return results


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider()