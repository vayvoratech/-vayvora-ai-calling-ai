"""Optional live integration test with a real Redis Stack instance.

This test is automatically skipped if a live Redis instance is not reachable
on the configured host and port.
"""

import asyncio
import pytest
from src.config import Settings
from src.core.types import DomainType, RAGChunk, RAGQuery
from src.rag.embeddings import MockEmbeddingProvider
from src.rag.redis_client import RedisVectorStore
from src.rag.retriever import GroundedKnowledgeProvider


async def is_redis_reachable() -> bool:
    """Check if a real Redis server responds to ping."""
    store = RedisVectorStore()
    try:
        is_up = await store.health_check()
        await store.close()
        return is_up
    except Exception:
        return False


def redis_available_check() -> bool:
    """Synchronous runner for pytest skipif condition."""
    try:
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(is_redis_reachable())
        loop.close()
        return res
    except Exception:
        return False


@pytest.mark.skipif(
    not redis_available_check(),
    reason="Live Redis Stack server is not reachable on configured port. Skipping live integration test.",
)
@pytest.mark.asyncio
async def test_live_redis_stack_vector_search():
    """Live integration test: creates index, ingests document, and executes KNN search."""
    settings = Settings()
    store = RedisVectorStore(settings=settings)
    provider = GroundedKnowledgeProvider(
        vector_store=store,
        embedding_provider=MockEmbeddingProvider(dimension=settings.rag_embedding_dimension),
        in_memory=False,
    )

    # 1. Initialize Index
    await store.create_index(DomainType.EDUSAAS, drop_existing=False)

    # 2. Ingest Chunk
    chunk = RAGChunk(
        doc_id="edu:live_test:001",
        domain=DomainType.EDUSAAS,
        title="Live Test Course",
        content="This is a live integration test document in Redis Stack.",
        score=1.0,
        metadata={"category": "live_test"},
    )
    await provider.index_document(chunk)

    # 3. Search
    query = RAGQuery(
        domain=DomainType.EDUSAAS,
        query_text="live integration test document",
        top_k=2,
        relevance_threshold=0.5,
    )
    result = await provider.retrieve_grounded_context(query)

    assert result.service_unavailable is False
    assert result.knowledge_available is True
    assert len(result.chunks) >= 1
    assert "edu:live_test:001" in [c.doc_id for c in result.chunks]

    await store.close()
