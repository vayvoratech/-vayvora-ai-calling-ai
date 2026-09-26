"""Redis Stack vector store and index management.

Provides separate namespaces and search indexes for EduSaaS and Vayvora,
KNN cosine vector search, metadata filtering, hybrid search, and health checks.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
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
    """Redis Stack client supporting vector indexes and grounded document retrieval."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.redis_url = self.settings.redis_url
        self.edusaas_index = self.settings.redis_edusaas_index
        self.vayvora_index = self.settings.redis_vayvora_index
        self.dim = self.settings.rag_embedding_dimension
        self._client: Optional[redis.Redis] = None

    async def get_client(self) -> redis.Redis:
        """Initialize or return cached async Redis client."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Keep binary support for float32 vector blobs
            )
        return self._client

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Redis server is reachable."""
        try:
            client = await self.get_client()
            return await client.ping()
        except Exception as exc:
            logger.debug("Redis health check failed: %s", exc)
            return False

    def get_index_name(self, domain: DomainType) -> str:
        """Map domain to isolated index name."""
        if domain == DomainType.EDUSAAS:
            return self.edusaas_index
        elif domain == DomainType.VAYVORA:
            return self.vayvora_index
        return f"idx:{domain.value}_vdb"

    def get_doc_prefix(self, domain: DomainType) -> str:
        """Map domain to key namespace prefix."""
        return f"rag:{domain.value}:doc:"

    async def create_index(self, domain: DomainType, drop_existing: bool = False) -> bool:
        """Create RediSearch vector and metadata index for a given domain."""
        client = await self.get_client()
        index_name = self.get_index_name(domain)
        prefix = self.get_doc_prefix(domain)

        try:
            # Check if index already exists
            info = await client.ft(index_name).info()
            if info and not drop_existing:
                logger.info("Index '%s' already exists.", index_name)
                return True
            if drop_existing:
                logger.info("Dropping existing index '%s'...", index_name)
                await client.ft(index_name).dropindex(delete_documents=False)
        except Exception:
            # Index does not exist, proceed to create
            pass

        schema = (
            TagField("doc_id"),
            TagField("domain"),
            TagField("category"),
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
            await client.ft(index_name).create_index(fields=schema, definition=definition)
            logger.info("Successfully created index '%s' for domain '%s'.", index_name, domain.value)
            return True
        except Exception as exc:
            logger.error("Failed to create index '%s': %s", index_name, exc)
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
    ) -> bool:
        """Ingest or update a document chunk in Redis."""
        client = await self.get_client()
        key = f"{self.get_doc_prefix(domain)}{doc_id}"

        # Convert float vector to 32-bit float byte string
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()

        mapping = {
            b"doc_id": doc_id.encode("utf-8"),
            b"domain": domain.value.encode("utf-8"),
            b"category": category.encode("utf-8"),
            b"title": title.encode("utf-8"),
            b"content": content.encode("utf-8"),
            b"vector": vector_bytes,
            b"metadata": json.dumps(metadata or {}).encode("utf-8"),
        }

        await client.hset(key, mapping=mapping)
        return True

    async def search_vector(
        self,
        domain: DomainType,
        query_vector: List[float],
        top_k: int = 3,
        relevance_threshold: float = 0.0,
        category: Optional[str] = None,
    ) -> List[RAGChunk]:
        """Perform KNN vector similarity search with domain isolation and relevance cutoff."""
        client = await self.get_client()
        index_name = self.get_index_name(domain)

        # Build filter query enforcing strict domain isolation
        filter_expr = f"(@domain:{{{domain.value}}}"
        if category:
            filter_expr += f" @category:{{{category}}}"
        filter_expr += ")"

        # Redis KNN query format: (filter)=>[KNN top_k @vector $vec AS score]
        query_str = f"{filter_expr}=>[KNN {top_k} @vector $vec AS score]"
        query = (
            Query(query_str)
            .sort_by("score")
            .return_fields("doc_id", "domain", "category", "title", "content", "metadata", "score")
            .dialect(2)
        )

        query_bytes = np.array(query_vector, dtype=np.float32).tobytes()

        try:
            results = await client.ft(index_name).search(query, query_params={"vec": query_bytes})
        except Exception as exc:
            logger.error("Vector search failed on index '%s': %s", index_name, exc)
            return []

        chunks: List[RAGChunk] = []
        for doc in results.docs:
            try:
                # In COSINE distance: score is distance d in [0, 2]. Similarity s = 1.0 - d
                raw_dist = float(getattr(doc, "score", 1.0))
                similarity = round(max(0.0, 1.0 - raw_dist), 4)

                # Relevance threshold filter
                if similarity < relevance_threshold:
                    continue

                raw_meta = getattr(doc, "metadata", b"{}")
                meta_str = raw_meta.decode("utf-8") if isinstance(raw_meta, bytes) else str(raw_meta)
                metadata = json.loads(meta_str) if meta_str else {}

                doc_id_val = doc.doc_id.decode("utf-8") if isinstance(doc.doc_id, bytes) else str(doc.doc_id)
                title_val = doc.title.decode("utf-8") if isinstance(doc.title, bytes) else str(doc.title)
                content_val = doc.content.decode("utf-8") if isinstance(doc.content, bytes) else str(doc.content)

                chunks.append(
                    RAGChunk(
                        doc_id=doc_id_val,
                        domain=domain,
                        title=title_val,
                        content=content_val,
                        score=similarity,
                        metadata=metadata,
                    )
                )
            except Exception as parse_err:
                logger.warning("Error parsing retrieved document chunk: %s", parse_err)
                continue

        return chunks
