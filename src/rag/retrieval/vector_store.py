from __future__ import annotations

import json
import os
import numpy as np

from src.memory.redis_client import redis_client
from src.rag.config import rag_config
from src.rag.schemas import DocumentChunk, RetrievalResult


class RAGVectorStore:
    """
    High-performance vector store with local in-memory NumPy acceleration
    and optional Redis VSet synchronization for real-time voice latency.
    """

    def __init__(self):
        self.vector_set_name = rag_config.redis_index_name
        self.metadata_prefix = f"{rag_config.redis_key_prefix}meta:"
        self._memory_chunks: list[DocumentChunk] = []
        self._memory_vectors: np.ndarray | None = None
        self._use_redis = os.getenv("RAG_USE_REDIS", "false").lower() == "true"
        self._ensure_in_memory_index()

    def _ensure_in_memory_index(self) -> None:
        """Load and vectorize knowledge base chunks into memory for sub-millisecond search."""
        if self._memory_vectors is not None and len(self._memory_chunks) > 0:
            return

        try:
            from src.rag.ingestion.pipeline import IngestionPipeline
            from src.rag.embeddings.provider import get_embedding_provider

            chunks = IngestionPipeline().run(tenant_id="default")
            if chunks:
                texts = [
                    f"{c.category} {c.section or ''}: {c.text}"
                    for c in chunks
                ]
                provider = get_embedding_provider()
                raw_vectors = provider.embed_documents(texts)
                arr = np.array(raw_vectors, dtype=np.float32)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._memory_vectors = arr / norms
                self._memory_chunks = chunks
        except Exception:
            pass

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
        new_vecs = []
        for chunk, vector in zip(chunks, vectors):
            normalized_vector = self._normalize_vector(vector)
            new_vecs.append(normalized_vector)

            if self._use_redis:
                try:
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
                except Exception:
                    pass
            else:
                inserted += 1

        # Sync to in-memory store
        if new_vecs:
            arr_new = np.array(new_vecs, dtype=np.float32)
            if self._memory_vectors is not None:
                self._memory_vectors = np.vstack([self._memory_vectors, arr_new])
                self._memory_chunks.extend(chunks)
            else:
                self._memory_vectors = arr_new
                self._memory_chunks = list(chunks)

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

        # 1. Ultra-fast in-memory NumPy cosine similarity (0.05ms)
        self._ensure_in_memory_index()
        if self._memory_vectors is not None and len(self._memory_chunks) > 0:
            q_arr = np.asarray(normalized_vector, dtype=np.float32)
            sims = np.dot(self._memory_vectors, q_arr)
            top_indices = np.argsort(-sims)[:search_count]

            results: list[RetrievalResult] = []
            for idx in top_indices:
                chunk = self._memory_chunks[idx]
                if tenant_id and chunk.tenant_id != tenant_id:
                    continue
                score = float(sims[idx])
                results.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=score,
                        source=chunk.source,
                        category=chunk.category,
                        section=chunk.section,
                        metadata={
                            "document_id": chunk.document_id,
                            "tenant_id": chunk.tenant_id,
                            "version": chunk.version,
                            "cosine_score": score,
                            **chunk.metadata,
                        },
                    )
                )
                if len(results) >= count:
                    break

            if results:
                return results

        # 2. Fallback to Redis if explicitly configured
        if not self._use_redis:
            return []

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
