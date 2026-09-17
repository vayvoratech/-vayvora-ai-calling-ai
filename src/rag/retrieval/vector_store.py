from __future__ import annotations

import json
import numpy as np

from src.memory.redis_client import redis_client
from src.rag.config import rag_config
from src.rag.schemas import DocumentChunk, RetrievalResult


class RAGVectorStore:
    """
    Redis vector store implementing cosine similarity vector set (VSet) search.
    """

    def __init__(self):
        self.vector_set_name = rag_config.redis_index_name
        self.metadata_prefix = f"{rag_config.redis_key_prefix}meta:"

    def _metadata_key(self, chunk_id: str) -> str:
        return f"{self.metadata_prefix}{chunk_id}"

    def _normalize_vector(self, vector) -> list[float]:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("Embedding vector must be one-dimensional.")
        if len(array) != rag_config.embedding_dimension:
            raise ValueError(
                f"Dimension mismatch: expected {rag_config.embedding_dimension}, got {len(array)}"
            )

        norm = np.linalg.norm(array)
        if norm == 0:
            raise ValueError("Cannot search or store a zero vector.")
        return (array / norm).tolist()

    def create_index(self) -> bool:
        return True

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks and vectors must match.")

        inserted = 0
        for chunk, vector in zip(chunks, vectors):
            normalized_vector = self._normalize_vector(vector)
            result = redis_client.execute_command(
                "VADD",
                self.vector_set_name,
                "VALUES",
                str(len(normalized_vector)),
                *[str(val) for val in normalized_vector],
                chunk.chunk_id,
            )

            metadata = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "tenant_id": chunk.tenant_id,
                "text": chunk.text,
                "source": chunk.source,
                "category": chunk.category,
                "section": chunk.section or "",
                "version": chunk.version,
                "metadata": chunk.metadata,
            }

            redis_client.set(
                self._metadata_key(chunk.chunk_id),
                json.dumps(metadata, ensure_ascii=False),
            )

            if result:
                inserted += 1

        return inserted

    def search(
        self,
        query_vector,
        top_k: int | None = None,
        tenant_id: str | None = None,
    ) -> list[RetrievalResult]:
        normalized_vector = self._normalize_vector(query_vector)
        count = top_k or rag_config.top_k
        search_count = count * 3 if tenant_id else count

        try:
            raw_results = redis_client.execute_command(
                "VSIM",
                self.vector_set_name,
                "VALUES",
                str(len(normalized_vector)),
                *[str(value) for value in normalized_vector],
                "WITHSCORES",
                "COUNT",
                str(search_count),
            )
        except Exception:
            return []

        results: list[RetrievalResult] = []
        if isinstance(raw_results, dict):
            result_items = raw_results.items()
        elif raw_results:
            result_items = zip(raw_results[::2], raw_results[1::2])
        else:
            return []

        for chunk_id, score in result_items:
            if isinstance(chunk_id, bytes):
                chunk_id = chunk_id.decode("utf-8")

            score = float(score)
            metadata_raw = redis_client.get(self._metadata_key(chunk_id))
            if not metadata_raw:
                continue

            if isinstance(metadata_raw, bytes):
                metadata_raw = metadata_raw.decode("utf-8")

            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                continue

            if tenant_id and metadata.get("tenant_id") != tenant_id:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=metadata["chunk_id"],
                    text=metadata["text"],
                    score=score,
                    source=metadata["source"],
                    category=metadata["category"],
                    section=metadata.get("section"),
                    metadata={
                        "document_id": metadata["document_id"],
                        "tenant_id": metadata["tenant_id"],
                        "version": metadata.get("version", "1"),
                        **metadata.get("metadata", {}),
                    },
                )
            )

            if len(results) >= count:
                break

        return results

    def delete_chunk(self, chunk_id: str) -> bool:
        v_del = redis_client.execute_command("VREM", self.vector_set_name, chunk_id)
        m_del = redis_client.delete(self._metadata_key(chunk_id))
        return bool(v_del or m_del)

    def delete_document(self, document_id: str, tenant_id: str = "default") -> int:
        deleted = 0
        keys = redis_client.scan_iter(match=f"{self.metadata_prefix}*")

        for key in keys:
            metadata_raw = redis_client.get(key)
            if not metadata_raw:
                continue
            if isinstance(metadata_raw, bytes):
                metadata_raw = metadata_raw.decode("utf-8")

            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                continue

            if metadata.get("document_id") != document_id or metadata.get("tenant_id") != tenant_id:
                continue

            chunk_id = metadata.get("chunk_id")
            if chunk_id and self.delete_chunk(chunk_id):
                deleted += 1

        return deleted

    def delete_all(self, tenant_id: str | None = None) -> dict:
        keys = redis_client.scan_iter(match=f"{self.metadata_prefix}*")
        deleted_meta = 0
        for k in keys:
            deleted_meta += redis_client.delete(k)
        v_del = redis_client.delete(self.vector_set_name)
        return {"vector_set_deleted": bool(v_del), "metadata_deleted": deleted_meta}
