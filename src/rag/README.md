# Grounded RAG Architecture & Redis Stack Guide

This document describes the Retrieval-Augmented Generation (RAG) implementation for the Unified Conversational AI Voice Agent across the **EduSaaS** and **Vayvora** business domains.

---

## 1. Redis Stack Requirement
- **Requirement:** Redis Stack (or Redis with the `RediSearch` module version 2.6+).
- **Features Used:**
  - HNSW (Hierarchical Navigable Small World) vector indexing.
  - Cosine distance metric for dense semantic search.
  - Tag filtering for strict multi-tenant domain isolation (`@domain:{edusaas}`, `@domain:{vayvora}`).
  - Full-text indexing on document titles and contents.

---

## 2. Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `REDIS_HOST` | `localhost` | Redis server hostname or IP address |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Logical database index |
| `REDIS_PASSWORD` | *(None)* | Optional authentication password |
| `REDIS_EDUSAAS_INDEX` | `idx:edusaas_vdb` | Search index identifier for EduSaaS |
| `REDIS_VAYVORA_INDEX` | `idx:vayvora_vdb` | Search index identifier for Vayvora |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | FastEmbed ONNX local embedding model |
| `RAG_EMBEDDING_DIMENSION` | `384` | Dimensionality of embedding vectors |
| `RAG_CHUNK_SIZE` | `500` | Target maximum character length per chunk |
| `RAG_CHUNK_OVERLAP` | `50` | Character overlap between consecutive chunks |
| `RAG_TOP_K` | `3` | Maximum number of chunks retrieved per query |
| `RAG_RELEVANCE_THRESHOLD`| `0.70` | Cosine similarity cutoff for grounding |

---

## 3. Creating Indexes

Indexes can be initialized via the `RedisVectorStore`:
```python
import asyncio
from src.core.types import DomainType
from src.rag.redis_client import RedisVectorStore

async def init_indexes():
    store = RedisVectorStore()
    await store.create_index(DomainType.EDUSAAS)
    await store.create_index(DomainType.VAYVORA)
    await store.close()

asyncio.run(init_indexes())
```

---

## 4. Ingesting Documents

To ingest all Markdown documents from `knowledge_base/edusaas/` and `knowledge_base/vayvora/`:
```python
import asyncio
from src.rag.retriever import GroundedKnowledgeProvider
from src.rag.ingestion import KnowledgeIngestionPipeline

async def run_ingestion():
    provider = GroundedKnowledgeProvider()
    pipeline = KnowledgeIngestionPipeline(knowledge_provider=provider)
    summary = await pipeline.ingest_all()
    print("Ingestion summary:", summary)

asyncio.run(run_ingestion())
```

---

## 5. Performing a Test Query

```python
import asyncio
from src.core.types import DomainType, RAGQuery
from src.rag.retriever import GroundedKnowledgeProvider

async def test_query():
    provider = GroundedKnowledgeProvider()
    query = RAGQuery(
        domain=DomainType.EDUSAAS,
        query_text="What are the prerequisites for technical courses?",
        top_k=3,
        relevance_threshold=0.70,
    )
    result = await provider.retrieve_grounded_context(query)
    print("Knowledge available:", result.knowledge_available)
    print("Formatted reference context:\n", result.formatted_context)

asyncio.run(test_query())
```

---

## 6. Domain Isolation Architecture
- **Separate Namespaces:**
  - EduSaaS documents use key prefix: `rag:edusaas:doc:<doc_id>`
  - Vayvora documents use key prefix: `rag:vayvora:doc:<doc_id>`
- **Isolated Indexes:**
  - EduSaaS queries only target `idx:edusaas_vdb`.
  - Vayvora queries only target `idx:vayvora_vdb`.
- **Query Filter Enforcement:** Every search enforces `@domain:{<domain_name>}`, guaranteeing zero cross-domain leakage even if contents have high semantic similarity.

---

## 7. Relevance Threshold & Grounding Guardrails
- **Cosine Distance Conversion:** Redis returns cosine distance $d \in [0, 2]$. We convert to cosine similarity $s = \max(0.0, 1.0 - d)$.
- **Relevance Cutoff:** Only chunks with $s \ge \text{RAG\_RELEVANCE\_THRESHOLD}$ (default: `0.70`) are returned.
- **Unavailable Information Policy:** If no chunk passes the threshold, the system sets `knowledge_available = False` and explicitly informs the caller that verified details are not on file, preventing hallucination.
- **Untrusted Reference Data:** Retrieved chunks are injected with `<verified_reference_data untrusted="true">` tags, instructing the model to treat documents strictly as reference content and never execute instructions found inside.

---

## 8. Running Unit Tests (Offline / Mocked)
All core RAG unit tests run completely offline without requiring a live Redis server or internet connection:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag.py tests/test_chunker.py
```

---

## 9. Running Optional Live Redis Integration Tests
When a local or remote Redis Stack instance is running, live integration tests can be run:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_redis_live.py
```
*(If Redis is not running on localhost:6379, this test is safely skipped automatically).*
