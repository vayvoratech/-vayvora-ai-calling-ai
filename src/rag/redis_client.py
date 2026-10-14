"""Redis Stack vector store and index management for JB Infra RAG."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.config import Settings, get_settings
from src.core.types import DomainType, RAGChunk, RAGQuery
from src.logging import get_logger


logger = get_logger("rag.redis")


class RedisVectorStore:
    """Redis Stack client for JB Infra vector search and grounded retrieval."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

        self.redis_url = self.settings.redis_url

        # JB Infra MVP uses a single general RAG index.
        self.index_name = getattr(
            self.settings,
            "redis_rag_index",
            "idx:general_vdb",
        )

        # Keep compatibility with the existing Settings configuration
        # if redis_index_name is already defined there.
        configured_index = getattr(
            self.settings,
            "redis_index_name",
            None,
        )

        if configured_index:
            self.index_name = configured_index

        self.dim = self.settings.rag_embedding_dimension
        self._client: Optional[redis.Redis] = None

    async def get_client(self) -> redis.Redis:
        """Initialize or return the cached async Redis client."""

        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=False,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )

        return self._client

    async def close(self) -> None:
        """Close the Redis connection."""

        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check whether the Redis server is reachable."""

        try:
            client = await self.get_client()
            result = await asyncio.wait_for(
                client.ping(),
                timeout=1.0,
            )
            return bool(result)

        except Exception as exc:
            logger.debug(
                "Redis health check failed: %s",
                exc,
            )
            return False

    def get_index_name(self, domain: DomainType) -> str:
        """Return the Redis index used by the JB Infra RAG system."""

        # The MVP intentionally uses one isolated general index.
        return self.index_name

    def get_doc_prefix(self, domain: DomainType) -> str:
        """Return the Redis document namespace."""

        return "rag:general:doc:"

    async def create_index(
        self,
        domain: DomainType,
        drop_existing: bool = False,
    ) -> bool:
        """Create the Redis vector and metadata index."""

        client = await self.get_client()

        index_name = self.get_index_name(domain)
        prefix = self.get_doc_prefix(domain)

        try:
            # Check whether the index already exists.
            info = await client.ft(index_name).info()

            if info and not drop_existing:
                logger.info(
                    "Index '%s' already exists.",
                    index_name,
                )
                return True

            if drop_existing:
                logger.info(
                    "Dropping existing index '%s'...",
                    index_name,
                )

                await client.ft(index_name).dropindex(
                    delete_documents=False,
                )

        except Exception:
            # Index does not exist, so continue with creation.
            pass

        schema = (
            TagField("doc_id"),
            TagField("domain"),
            TagField("category"),
            TagField("document_type"),
            TextField("title"),
            TextField("content"),
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.dim,
                    "DISTANCE_METRIC": "COSINE",
                    "INITIAL_CAP": 500,
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )

        definition = IndexDefinition(
            prefix=[prefix],
            index_type=IndexType.HASH,
        )

        try:
            await client.ft(index_name).create_index(
                fields=schema,
                definition=definition,
            )

            logger.info(
                "Successfully created index '%s'.",
                index_name,
            )

            return True

        except Exception as exc:
            logger.error(
                "Failed to create index '%s': %s",
                index_name,
                exc,
            )
            raise

    async def index_document(
        self,
        domain: DomainType,
        doc_id: str,
        title: str,
        content: str,
        category: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        document_type: str = "general",
    ) -> bool:
        """Ingest or update a document chunk in Redis."""

        client = await self.get_client()

        key = f"{self.get_doc_prefix(domain)}{doc_id}"

        vector_bytes = np.array(
            vector,
            dtype=np.float32,
        ).tobytes()

        meta = metadata or {}

        doc_type = (
            meta.get("document_type")
            or document_type
        )

        mapping = {
            b"doc_id": doc_id.encode("utf-8"),
            b"domain": DomainType.GENERAL.value.encode("utf-8"),
            b"category": category.encode("utf-8"),
            b"document_type": doc_type.encode("utf-8"),
            b"title": title.encode("utf-8"),
            b"content": content.encode("utf-8"),
            b"vector": vector_bytes,
            b"metadata": json.dumps(meta).encode("utf-8"),
        }

        await client.hset(
            key,
            mapping=mapping,
        )

        return True

    async def search_vector(
        self,
        domain: DomainType,
        query_vector: List[float],
        top_k: int = 3,
        relevance_threshold: float = 0.0,
        category: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> List[RAGChunk]:
        """Perform KNN vector search with strict general-domain isolation."""

        client = await self.get_client()

        index_name = self.get_index_name(domain)

        # The JB Infra MVP stores all approved knowledge in the
        # general domain namespace.
        filter_expr = "(@domain:{general}"

        if category:
            filter_expr += f" @category:{{{category}}}"

        if document_type:
            filter_expr += f" @document_type:{{{document_type}}}"

        filter_expr += ")"

        query_str = (
            f"{filter_expr}"
            f"=>[KNN {top_k} @vector $vec AS score]"
        )

        query = (
            Query(query_str)
            .sort_by("score")
            .return_fields(
                "doc_id",
                "domain",
                "category",
                "title",
                "content",
                "metadata",
                "score",
            )
            .dialect(2)
        )

        query_bytes = np.array(
            query_vector,
            dtype=np.float32,
        ).tobytes()

        try:
            results = await client.ft(index_name).search(
                query,
                query_params={
                    "vec": query_bytes,
                },
            )

        except Exception as exc:
            logger.error(
                "Vector search failed on index '%s': %s",
                index_name,
                exc,
            )
            return []

        chunks: List[RAGChunk] = []

        for doc in results.docs:
            try:
                # Redis COSINE returns distance.
                # Convert distance to similarity.
                raw_dist = float(
                    getattr(
                        doc,
                        "score",
                        1.0,
                    )
                )

                similarity = round(
                    max(
                        0.0,
                        1.0 - raw_dist,
                    ),
                    4,
                )

                if similarity < relevance_threshold:
                    continue

                raw_meta = getattr(
                    doc,
                    "metadata",
                    b"{}",
                )

                meta_str = (
                    raw_meta.decode("utf-8")
                    if isinstance(raw_meta, bytes)
                    else str(raw_meta)
                )

                metadata = (
                    json.loads(meta_str)
                    if meta_str
                    else {}
                )

                doc_id_val = (
                    doc.doc_id.decode("utf-8")
                    if isinstance(doc.doc_id, bytes)
                    else str(doc.doc_id)
                )

                title_val = (
                    doc.title.decode("utf-8")
                    if isinstance(doc.title, bytes)
                    else str(doc.title)
                )

                content_val = (
                    doc.content.decode("utf-8")
                    if isinstance(doc.content, bytes)
                    else str(doc.content)
                )

                chunks.append(
                    RAGChunk(
                        doc_id=doc_id_val,
                        domain=DomainType.GENERAL,
                        title=title_val,
                        content=content_val,
                        score=similarity,
                        metadata=metadata,
                    )
                )

            except Exception as parse_err:
                logger.warning(
                    "Error parsing retrieved document chunk: %s",
                    parse_err,
                )
                continue

        return chunks

    async def get_documents_by_type(
        self,
        domain: DomainType,
        document_type: str,
        limit: int = 5,
    ) -> List[RAGChunk]:
        """Retrieve JB Infra documents by document type."""

        client = await self.get_client()

        index_name = self.get_index_name(domain)

        query_str = (
            f"(@domain:{{general}} "
            f"@document_type:{{{document_type}}})"
        )

        query = (
            Query(query_str)
            .return_fields(
                "doc_id",
                "domain",
                "category",
                "document_type",
                "title",
                "content",
                "metadata",
            )
            .paging(0, limit)
            .dialect(2)
        )

        try:
            results = await client.ft(index_name).search(
                query
            )

        except Exception as exc:
            logger.error(
                "Metadata search by document_type failed "
                "on index '%s': %s",
                index_name,
                exc,
            )
            return []

        chunks: List[RAGChunk] = []

        for doc in results.docs:
            try:
                raw_meta = getattr(
                    doc,
                    "metadata",
                    b"{}",
                )

                meta_str = (
                    raw_meta.decode("utf-8")
                    if isinstance(raw_meta, bytes)
                    else str(raw_meta)
                )

                metadata = (
                    json.loads(meta_str)
                    if meta_str
                    else {}
                )

                doc_id_val = (
                    doc.doc_id.decode("utf-8")
                    if isinstance(doc.doc_id, bytes)
                    else str(doc.doc_id)
                )

                title_val = (
                    doc.title.decode("utf-8")
                    if isinstance(doc.title, bytes)
                    else str(doc.title)
                )

                content_val = (
                    doc.content.decode("utf-8")
                    if isinstance(doc.content, bytes)
                    else str(doc.content)
                )

                chunks.append(
                    RAGChunk(
                        doc_id=doc_id_val,
                        domain=DomainType.GENERAL,
                        title=title_val,
                        content=content_val,
                        score=1.0,
                        metadata=metadata,
                    )
                )

            except Exception as parse_err:
                logger.warning(
                    "Error parsing retrieved document chunk: %s",
                    parse_err,
                )
                continue

        return chunks