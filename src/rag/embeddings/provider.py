import os
from functools import lru_cache
import numpy as np
import requests
from dotenv import load_dotenv

from src.rag.config import rag_config

load_dotenv()


class EmbeddingProvider:
    """
    Zero-RAM Cloud Embedding Provider using Hugging Face's serverless router.
    Explicitly targets the feature-extraction pipeline for raw dense embeddings.
    """

    def __init__(self, model_name: str | None = None):
        self.model = "sentence-transformers/all-MiniLM-L6-v2"
        # Explicit pipeline URL to prevent HF from defaulting to sentence-similarity
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model}/pipeline/feature-extraction"

        token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.dimension = getattr(rag_config, "embedding_dimension", 384)
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        """Unit L2-normalization for cosine distance computations."""
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def _query(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=25)
        if not resp.ok:
            raise RuntimeError(f"Hugging Face Router API error [{resp.status_code}]: {resp.text}")

        data = resp.json()

        # Handle nested token embeddings vs pooled sentence embeddings
        results = []
        if isinstance(data, list):
            for item in data:
                # If 2D (sequence_length, hidden_dim), mean-pool over tokens
                if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
                    arr = np.mean(np.array(item, dtype=np.float32), axis=0)
                    results.append(arr.tolist())
                elif isinstance(item, list) and (len(item) == 0 or isinstance(item[0], (float, int))):
                    results.append(item)
                else:
                    results.append(item)
        return results

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        cleaned = text.strip()
        if cleaned in self._cache:
            return self._cache[cleaned]

        res = self._query([cleaned])
        result = self._normalize(res[0])

        if len(self._cache) < 4096:
            self._cache[cleaned] = result
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

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
            # Batch process in chunks of 16
            chunk_size = 16
            for i in range(0, len(to_encode), chunk_size):
                chunk_texts = to_encode[i:i + chunk_size]
                chunk_indices = indices[i:i + chunk_size]

                raw_vectors = self._query(chunk_texts)
                for idx, raw_vec in zip(chunk_indices, raw_vectors):
                    normalized_vec = self._normalize(raw_vec)
                    results[idx] = normalized_vec
                    cached_text = texts[idx].strip()
                    if len(self._cache) < 4096:
                        self._cache[cached_text] = normalized_vec

        return results


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider()