"""Vector storage and similarity search for the JB Infra RAG MVP."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.rag.config import rag_config
from src.rag.models import DocumentChunk


class RAGVectorStore:
    """Lightweight vector store for the JB Infra demo.

    The MVP keeps vectors in memory for fast local testing. Redis support
    can be enabled later without changing the retrieval interface.
    """

    def __init__(
        self,
        tenant_id: str = "default",
    ) -> None:
        self.tenant_id = tenant_id
        self.vector_dimension = rag_config.embedding_dimension

        self._vectors: Dict[str, np.ndarray] = {}
        self._chunks: Dict[str, DocumentChunk] = {}

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        vectors: List[List[float]],
    ) -> int:
        """Add document chunks and their embeddings."""

        if len(chunks) != len(vectors):
            raise ValueError(
                "chunks and vectors must have the same length"
            )

        added = 0

        for chunk, vector in zip(
            chunks,
            vectors,
        ):
            if chunk.tenant_id != self.tenant_id:
                continue

            normalized = self._normalize_vector(
                vector
            )

            self._vectors[chunk.chunk_id] = normalized
            self._chunks[chunk.chunk_id] = chunk

            added += 1

        return added

    def add_chunk(
        self,
        chunk: DocumentChunk,
        vector: List[float],
    ) -> bool:
        """Add a single chunk and embedding."""

        return (
            self.add_chunks(
                chunks=[chunk],
                vectors=[vector],
            )
            == 1
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        tenant_id: Optional[str] = None,
    ) -> List[tuple[DocumentChunk, float]]:
        """Return the most similar chunks using cosine similarity."""

        if not self._vectors:
            return []

        query = self._normalize_vector(
            query_vector
        )

        target_tenant = (
            tenant_id
            if tenant_id is not None
            else self.tenant_id
        )

        scored = []

        for chunk_id, vector in self._vectors.items():
            chunk = self._chunks.get(chunk_id)

            if chunk is None:
                continue

            if chunk.tenant_id != target_tenant:
                continue

            score = float(
                np.dot(
                    query,
                    vector,
                )
            )

            scored.append(
                (
                    chunk,
                    score,
                )
            )

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored[:top_k]

    def get_chunk(
        self,
        chunk_id: str,
    ) -> Optional[DocumentChunk]:
        """Return a stored chunk by ID."""

        return self._chunks.get(chunk_id)

    def delete_chunk(
        self,
        chunk_id: str,
    ) -> bool:
        """Delete one chunk."""

        existed = (
            chunk_id in self._chunks
            or chunk_id in self._vectors
        )

        self._chunks.pop(
            chunk_id,
            None,
        )

        self._vectors.pop(
            chunk_id,
            None,
        )

        return existed

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        """Delete all chunks belonging to a document."""

        chunk_ids = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if (
                chunk.document_id
                == document_id
            )
            and (
                chunk.tenant_id
                == self.tenant_id
            )
        ]

        for chunk_id in chunk_ids:
            self.delete_chunk(chunk_id)

        return len(chunk_ids)

    def delete_all(self) -> int:
        """Delete all vectors for this tenant."""

        chunk_ids = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.tenant_id == self.tenant_id
        ]

        for chunk_id in chunk_ids:
            self.delete_chunk(chunk_id)

        return len(chunk_ids)

    def count(self) -> int:
        """Return the number of stored chunks."""

        return sum(
            1
            for chunk in self._chunks.values()
            if chunk.tenant_id == self.tenant_id
        )

    def health_check(self) -> bool:
        """Return whether the local vector store is available."""

        return True

    @staticmethod
    def _normalize_vector(
        vector: List[float],
    ) -> np.ndarray:
        """Validate and L2-normalize a vector."""

        array = np.asarray(
            vector,
            dtype=np.float32,
        )

        if array.ndim != 1:
            raise ValueError(
                "Embedding vector must be one-dimensional"
            )

        if len(array) != rag_config.embedding_dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {rag_config.embedding_dimension}, "
                f"received {len(array)}"
            )

        norm = np.linalg.norm(array)

        if norm == 0:
            return array

        return array / norm


__all__ = [
    "RAGVectorStore",
]