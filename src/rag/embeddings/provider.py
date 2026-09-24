import os
from functools import lru_cache
import numpy as np
import requests
from dotenv import load_dotenv

from src.rag.config import rag_config

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required.")


class EmbeddingProvider:
    """
    Zero-RAM Cloud Embedding Provider using direct REST API calls.
    Avoids SDK v1beta naming bugs and runs with negligible memory overhead.
    """

    def __init__(self, model_name: str = "text-embedding-004"):
        # Strip prefixes
        self.model_name = model_name.replace("models/", "").strip()
        self.api_key = api_key
        # Target the stable API endpoint
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents?key={self.api_key}"
        self.single_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"
        
        self.dimension = getattr(rag_config, "embedding_dimension", 768)
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
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

        payload = {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": cleaned}]}
        }

        resp = requests.post(self.single_url, json=payload, timeout=10)
        
        # Fallback to embedding-001 if the project hasn't activated text-embedding-004
        if resp.status_code == 404:
            fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={self.api_key}"
            payload["model"] = "models/embedding-001"
            resp = requests.post(fallback_url, json=payload, timeout=10)

        resp.raise_for_status()
        raw_vector = resp.json()["embedding"]["values"]
        result = self._normalize(raw_vector)

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
            requests_body = [
                {
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": t}]}
                }
                for t in to_encode
            ]

            resp = requests.post(self.url, json={"requests": requests_body}, timeout=15)

            # Fallback if 404
            if resp.status_code == 404:
                fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:batchEmbedContents?key={self.api_key}"
                for r in requests_body:
                    r["model"] = "models/embedding-001"
                resp = requests.post(fallback_url, json={"requests": requests_body}, timeout=15)

            resp.raise_for_status()
            data = resp.json()

            for idx, item in zip(indices, data["embeddings"]):
                normalized_vec = self._normalize(item["values"])
                results[idx] = normalized_vec
                cached_text = texts[idx].strip()
                if len(self._cache) < 4096:
                    self._cache[cached_text] = normalized_vec

        return results


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider()