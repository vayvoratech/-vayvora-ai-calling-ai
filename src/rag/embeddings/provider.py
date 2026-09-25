from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.config import rag_config


class EmbeddingProvider:
    """
    Centralized embedding provider for the RAG system.

    The embedding model is loaded once and reused for the lifetime
    of the process.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = (
            model_name or rag_config.embedding_model
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        self.dimension = self.model.get_embedding_dimension()

        if self.dimension != rag_config.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"model={self.dimension}, "
                f"configured={rag_config.embedding_dimension}"
            )

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.astype(np.float32).tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        if any(
            not text or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Document list contains empty text."
            )

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        return vectors.astype(np.float32).tolist()


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """
    Return a singleton embedding provider.

    Prevents loading the transformer model repeatedly.
    """
    return EmbeddingProvider()