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
    Zero-RAM Cloud Embedding Provider using direct REST API.
    Bypasses SDK model resolution bugs and stays within Render 512MB limits.
    """

    def __init__(self, model_name: str | None = None):
        self.api_key = api_key
        # Target Google's stable v1beta REST endpoints
        self.embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
        self.batch_url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={self.api_key}"
        
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
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": cleaned}]}
        }

        resp = requests.post(self.embed_url, json=payload, timeout=10)
        
        # Fallback to embedding-001 if text-embedding-004 is restricted on the key
        if resp.status_code == 404:
            alt_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={self.api_key}"
            payload["model"] = "models/embedding-001"
            resp = requests.post(alt_url, json=payload, timeout=10)

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
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": t}]}
                }
                for t in to_encode
            ]

            resp = requests.post(self.batch_url, json={"requests": requests_body}, timeout=20)

            # Fallback for projects where text-embedding-004 is unavailable
            if resp.status_code == 404:
                alt_batch_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:batchEmbedContents?key={self.api_key}"
                for r in requests_body:
                    r["model"] = "models/embedding-001"
                resp = requests.post(alt_batch_url, json={"requests": requests_body}, timeout=20)

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