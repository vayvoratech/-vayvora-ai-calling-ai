import os
from functools import lru_cache
import numpy as np
from dotenv import load_dotenv
from google import genai

from src.rag.config import rag_config

load_dotenv()

# Resolve API Key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required.")


class EmbeddingProvider:
    """
    Zero-RAM Cloud Embedding Provider using Google GenAI SDK.
    Eliminates all local neural weights, HF Hub downloads, and memory bloat.
    """

    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        # Native lightweight Google GenAI client (pure HTTP)
        self.client = genai.Client(api_key=api_key)
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

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=cleaned,
        )
        # response.embeddings[0].values contains the 768-dim vector
        raw_vector = response.embeddings[0].values
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
            # Batch embedding via native GenAI client
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=to_encode,
            )
            for idx, embedding in zip(indices, response.embeddings):
                normalized_vec = self._normalize(embedding.values)
                results[idx] = normalized_vec
                cached_text = texts[idx].strip()
                if len(self._cache) < 4096:
                    self._cache[cached_text] = normalized_vec

        return results


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider()