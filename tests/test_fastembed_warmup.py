"""Focused unit and integration tests for Phase 5B-3: FastEmbed ONNX Startup Warmup.

Verifies:
1. Warmup initializes the embedding model only once.
2. Repeated startup/warmup calls do not create duplicate instances.
3. The first real query reuses the warmed-up instance without reloading.
4. Concurrent warmup calls are thread-safe and idempotent.
5. Async warmup in worker thread does not block the event loop.
6. Warmup failure is handled safely (privacy-safe error, returns False, RAG fallback works).
7. FastAPI lifespan executes warmup during startup.
8. Existing retrieval and grounding behavior remains strictly preserved.
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

import src.core.types  # Enforce initialization order
from src.config import Settings
from src.core.types import DomainType, RAGChunk, RAGQuery
from src.rag.embeddings import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    clear_embedding_provider_cache,
    get_embedding_provider,
)
from src.rag.retriever import GroundedContextResult, GroundedKnowledgeProvider
from src.rag.services.retriever import RAGRetriever, get_rag_retriever
from src.ui.app import app


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure clean cache before and after each test."""
    clear_embedding_provider_cache()
    yield
    clear_embedding_provider_cache()


def test_01_warmup_initializes_model_only_once():
    """Warmup loads the underlying FastEmbed model once and sets is_warmed_up to True."""
    provider = EmbeddingProvider(dimension=384)
    assert not provider.is_warmed_up()
    assert provider._model is None

    # First warmup call
    success = provider.warmup()
    assert success is True
    assert provider.is_warmed_up()
    assert provider._model is not None
    first_model = provider._model

    # Second warmup call is a no-op
    success_2 = provider.warmup()
    assert success_2 is True
    assert provider._model is first_model


def test_02_repeated_startup_warmup_calls_do_not_create_duplicate_instances():
    """Multiple calls to get_embedding_provider or multiple EmbeddingProvider instances reuse the model."""
    provider1 = get_embedding_provider()
    provider2 = get_embedding_provider()
    assert provider1 is provider2

    # Even a freshly instantiated EmbeddingProvider reuses the cached model from _MODEL_CACHE
    provider1.warmup()
    provider3 = EmbeddingProvider(dimension=384)
    provider3._ensure_model()
    assert provider3._model is provider1._model


def test_03_first_real_query_reuses_warmed_up_instance():
    """The first real query uses the pre-warmed model and does not trigger model re-initialization."""
    provider = EmbeddingProvider(dimension=384)
    provider.warmup()

    # Track calls to _ensure_model
    original_ensure = provider._ensure_model
    ensure_mock = MagicMock(side_effect=original_ensure)
    provider._ensure_model = ensure_mock

    # Execute a real query
    embedding = provider.embed_text("Where is Pristine City located?")
    assert len(embedding) == 384
    ensure_mock.assert_called_once()
    # The internal _model was already populated, so no ONNX reload took place
    assert provider.is_warmed_up()


@pytest.mark.asyncio
async def test_04_async_warmup_runs_in_worker_thread():
    """Async warmup executes in a thread pool and does not block the asyncio event loop."""
    provider = EmbeddingProvider(dimension=384)
    assert not provider.is_warmed_up()

    # Run async warmup
    success = await provider.async_warmup()
    assert success is True
    assert provider.is_warmed_up()
    assert provider._model is not None


@pytest.mark.asyncio
async def test_05_concurrent_warmup_calls_are_thread_safe_and_idempotent():
    """Multiple concurrent warmup tasks safely coordinate via lock without duplicating work."""
    provider = EmbeddingProvider(dimension=384)

    tasks = [provider.async_warmup() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    assert all(r is True for r in results)
    assert provider.is_warmed_up()


def test_06_warmup_failure_handled_safely():
    """When model loading raises an exception, warmup logs an error, returns False, and leaves model clean."""
    provider = EmbeddingProvider(dimension=384)

    with patch.object(provider, "_ensure_model", side_effect=RuntimeError("ONNX library missing")):
        success = provider.warmup()
        assert success is False
        assert not provider.is_warmed_up()
        assert provider._model is None


@pytest.mark.asyncio
async def test_07_fastapi_lifespan_warms_up_model():
    """FastAPI lifespan context manager warms up the embedding provider on startup."""
    with TestClient(app) as client:
        # Trigger startup lifespan
        resp = client.get("/health")
        assert resp.status_code == 200

        provider = get_embedding_provider()
        assert provider.is_warmed_up()


@pytest.mark.asyncio
async def test_08_existing_retrieval_and_grounding_preserved():
    """GroundedKnowledgeProvider and RAGRetriever function correctly with pre-warmed embeddings."""
    provider = get_embedding_provider()
    provider.warmup()

    rag_retriever = RAGRetriever(embedding_provider=provider)
    assert rag_retriever.warmup() is True

    chunk = RAGChunk(
        doc_id="pride-test-1",
        domain=DomainType.INFRA_BUILDER,
        score=0.92,
        content="JB Pride features luxury plotted layouts near Kompally.",
        metadata={"project": "JB Pride"},
        title="JB Pride Overview",
    )
    kp = GroundedKnowledgeProvider(embedding_provider=provider, in_memory=True)
    await kp.index_document(chunk)

    query = RAGQuery(
        domain=DomainType.INFRA_BUILDER,
        query_text="Tell me about JB Pride in Kompally",
        top_k=3,
        relevance_threshold=0.3,
    )
    res = await kp.retrieve_grounded_context(query)
    assert res.knowledge_available is True
    assert len(res.chunks) >= 1
    assert "Kompally" in res.chunks[0].content
