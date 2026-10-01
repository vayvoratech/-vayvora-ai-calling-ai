"""Vector store with NumPy cosine similarity acceleration and Redis integration."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.logging import get_logger
from src.rag.config import rag_config
from src.rag.models import DocumentChunk
from src.rag.schemas import RetrievalResult

logger = get_logger("rag.vector_store")


class RAGVectorStore:
    """
    High-performance vector store with local in-memory NumPy acceleration
    and optional Redis synchronization for sub-millisecond voice latency.
    """

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id
        self.vector_set_name = rag_config.redis_index_name
        self.metadata_prefix = f"{rag_config.redis_key_prefix}meta:"
        self._memory_chunks: List[DocumentChunk] = []
        self._memory_vectors: Optional[np.ndarray] = None
        self._use_redis = os.getenv("RAG_USE_REDIS", "false").lower() == "true"
        self._redis_client = None

    def _get_sync_redis(self):
        if self._redis_client is None:
            try:
                import redis
                self._redis_client = redis.from_url(
                    rag_config.redis_url,
                    decode_responses=False,
                )
            except Exception as e:
                logger.debug("Redis sync client init failed: %s", e)
                self._redis_client = False
        return self._redis_client if self._redis_client is not False else None

    def _metadata_key(self, chunk_id: str) -> str:
        return f"{self.metadata_prefix}{chunk_id}"

    def _normalize_vector(self, vector) -> List[float]:
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
        chunks: List[DocumentChunk],
        vectors: List[List[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks and vectors must match.")

        inserted = 0
        new_vecs = []
        for chunk, vector in zip(chunks, vectors):
            normalized_vector = self._normalize_vector(vector)
            new_vecs.append(normalized_vector)

            client = self._get_sync_redis()
            if self._use_redis and client:
                try:
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
                    client.set(
                        self._metadata_key(chunk.chunk_id),
                        json.dumps(metadata, ensure_ascii=False),
                    )
                    inserted += 1
                except Exception as exc:
                    logger.debug("Redis chunk sync failed: %s", exc)
            else:
                inserted += 1

        # Synchronize to local in-memory NumPy matrix
        if new_vecs:
            arr_new = np.array(new_vecs, dtype=np.float32)
            if self._memory_vectors is not None and len(self._memory_chunks) > 0:
                self._memory_vectors = np.vstack([self._memory_vectors, arr_new])
                self._memory_chunks.extend(chunks)
            else:
                self._memory_vectors = arr_new
                self._memory_chunks = list(chunks)

        return inserted

    def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        normalized_vector = self._normalize_vector(query_vector)
        count = top_k or rag_config.top_k
        search_count = count * 3 if tenant_id else count

        # In-memory NumPy cosine similarity (ultra-fast, <0.05ms)
        if self._memory_vectors is not None and len(self._memory_chunks) > 0:
            q_arr = np.asarray(normalized_vector, dtype=np.float32)
            sims = np.dot(self._memory_vectors, q_arr)
            top_indices = np.argsort(-sims)[:search_count]

            results: List[RetrievalResult] = []
            for idx in top_indices:
                chunk = self._memory_chunks[idx]
                if tenant_id and tenant_id not in ("default", "all") and chunk.tenant_id != tenant_id:
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

            return results

        return []

    def delete_chunk(self, chunk_id: str) -> bool:
        client = self._get_sync_redis()
        m_del = client.delete(self._metadata_key(chunk_id)) if client else 0

        # Purge from memory
        new_chunks = []
        keep_indices = []
        for idx, chunk in enumerate(self._memory_chunks):
            if chunk.chunk_id != chunk_id:
                new_chunks.append(chunk)
                keep_indices.append(idx)

        self._memory_chunks = new_chunks
        if self._memory_vectors is not None:
            if keep_indices:
                self._memory_vectors = self._memory_vectors[keep_indices]
            else:
                self._memory_vectors = None

        return bool(m_del or (len(self._memory_chunks) != len(new_chunks)))

    def delete_document(self, document_id: str, tenant_id: str = "default") -> int:
        deleted = 0
        client = self._get_sync_redis()
        if client:
            keys = client.scan_iter(match=f"{self.metadata_prefix}*")
            for key in keys:
                raw = client.get(key)
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    meta = json.loads(raw)
                    if meta.get("document_id") == document_id and (
                        tenant_id in ("default", "all") or meta.get("tenant_id") == tenant_id
                    ):
                        client.delete(key)
                        deleted += 1
                except Exception:
                    continue

        # Purge from memory
        if self._memory_chunks:
            new_chunks = []
            keep_indices = []
            for idx, c in enumerate(self._memory_chunks):
                match_doc = (c.document_id == document_id or c.source == document_id)
                match_tenant = (tenant_id in ("default", "all") or c.tenant_id == tenant_id)
                if match_doc and match_tenant:
                    deleted += 1
                else:
                    new_chunks.append(c)
                    keep_indices.append(idx)

            self._memory_chunks = new_chunks
            if self._memory_vectors is not None:
                if keep_indices:
                    self._memory_vectors = self._memory_vectors[keep_indices]
                else:
                    self._memory_vectors = None

        return deleted

    def delete_all(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        deleted_meta = 0
        client = self._get_sync_redis()
        if client:
            keys = client.scan_iter(match=f"{self.metadata_prefix}*")
            for k in keys:
                deleted_meta += client.delete(k)
            v_del = client.delete(self.vector_set_name)
        else:
            v_del = 1

        self._memory_chunks = []
        self._memory_vectors = None
        return {"vector_set_deleted": bool(v_del), "metadata_deleted": deleted_meta}
