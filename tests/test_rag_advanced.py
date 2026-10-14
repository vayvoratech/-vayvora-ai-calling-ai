"""Comprehensive test suite covering all migrated RAG components:
Chunking, Embeddings, Manifest, BM25, RRF, Reranker, Vector Store, Domain Isolation,
Thresholds, Fast Path, Course Recommendations, and Real Query Verification.
"""

import asyncio
from pathlib import Path
import pytest

from src.core.decision import ConversationalDecision
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.llm import MockLLMProvider
from src.core.types import ConversationStage, DomainType, RAGChunk, RAGQuery
from src.rag.config import rag_config
from src.rag.embeddings import EmbeddingProvider, MockEmbeddingProvider
from src.rag.ingestion.chunker import MarkdownChunker
from src.rag.ingestion.incremental import IncrementalIngestionService
from src.rag.ingestion.manifest import DocumentManifest, ManifestStore
from src.rag.models import DocumentChunk
from src.rag.retrieval.fusion import ReciprocalRankFusion
from src.rag.retrieval.hybrid_search import BM25Retriever, KeywordDocument
from src.rag.retrieval.keyword_index import KnowledgeBaseKeywordIndex
from src.rag.retrieval.reranker import Reranker
from src.rag.retrieval.vector_store import RAGVectorStore
from src.rag.retriever import GroundedContextResult, GroundedKnowledgeProvider
from src.rag.schemas import RAGResponse, RetrievalResult
from src.rag.services.fast_path import FAST_PATH_RULES, get_deterministic_fast_path
from src.rag.services.recommendation import CourseRecommendationService, PropertyRecommendationService
from src.rag.services.retriever import RAGRetriever
from src.state.manager import ConversationStateManager


# =========================================================================
# 1. CHUNKING
# =========================================================================
class TestMarkdownChunking:
    def test_chunking_preserves_hierarchy_and_context(self):
        chunker = MarkdownChunker(chunk_size=300, chunk_overlap=30)
        sample_md = """# Infra Builder Overview
## Development Architecture
Infra Builder provides premier residential properties and villas in Bangalore.
Our developments include Skyline Towers, Green Valley Villas, and Urban Heights.

## Security Compliance
All media streams are encrypted end-to-end with TLS and SRTP.
"""
        chunks = chunker.chunk_text(
            text=sample_md,
            domain_or_tenant="infra_builder",
            category="architecture",
            source_file="development_platform.md",
        )
        assert len(chunks) >= 2
        for c in chunks:
            assert c.tenant_id == "infra_builder"
            assert c.domain == DomainType.INFRA_BUILDER
            assert c.category == "architecture"
            assert "Infra Builder Overview" in c.text
            assert c.chunk_id.startswith("infra_builder:architecture:development_platform:")

    def test_chunk_size_validation(self):
        with pytest.raises(ValueError):
            MarkdownChunker(chunk_size=100, chunk_overlap=120)


# =========================================================================
# 2. EMBEDDINGS
# =========================================================================
class TestEmbeddingsSubsystem:
    def test_mock_embeddings_unit_norm_and_dimensions(self):
        p = MockEmbeddingProvider(dimension=384)
        assert p.dimension == 384
        vec = p.embed_text("Deep Learning for Speech Synthesis")
        assert len(vec) == 384
        norm = sum(x * x for x in vec) ** 0.5
        assert pytest.approx(norm, rel=1e-2) == 1.0

    def test_embedding_provider_caching_and_batch(self):
        p = EmbeddingProvider(dimension=384, cache_size=100)
        t1 = "Deterministic test text for LRU cache verification"
        v1 = p.embed_text(t1)
        assert t1 in p._cache
        v2 = p.embed_text(t1)
        assert v1 == v2

        batch = p.embed_batch([t1, "Second unique test sentence"])
        assert len(batch) == 2
        assert batch[0] == v1


# =========================================================================
# 3. MANIFEST & INCREMENTAL INGESTION
# =========================================================================
class TestManifestAndIncremental:
    def test_manifest_store_lifecycle(self, tmp_path):
        manifest_file = tmp_path / "test_manifest.json"
        store = ManifestStore(file_path=str(manifest_file))
        assert store.load() == {}

        doc_m = DocumentManifest(
            document_id="doc1",
            source="overview.md",
            tenant_id="edusaas",
            content_hash="abc123hash",
            chunk_count=5,
        )
        store.update(doc_m)

        loaded = store.load()
        assert "doc1" in loaded
        assert loaded["doc1"].content_hash == "abc123hash"
        assert loaded["doc1"].chunk_count == 5

        store.delete("doc1")
        assert store.load() == {}

    def test_incremental_service_detection(self, tmp_path):
        doc_dir = tmp_path / "edusaas"
        doc_dir.mkdir(parents=True)
        file1 = doc_dir / "sample.md"
        file1.write_text("# Sample\n\nSample content for incremental testing.", encoding="utf-8")

        manifest_file = tmp_path / ".manifest.json"
        store = ManifestStore(file_path=str(manifest_file))
        service = IncrementalIngestionService(
            tenant_id="edusaas",
            loader=None,
            chunker=MarkdownChunker(chunk_size=500, chunk_overlap=50),
            manifest=store,
        )
        service.loader.base_dir = tmp_path

        plan = service.prepare()
        assert len(plan["changed_documents"]) >= 1

        # Mark as ingested in manifest
        doc_item = plan["changed_documents"][0]
        service.update_manifest(
            doc_item["document"],
            doc_item["content_hash"],
            len(doc_item["chunks"]),
        )

        # Second check: document is now unchanged
        plan2 = service.prepare()
        assert len(plan2["unchanged_documents"]) >= 1
        assert len(plan2["changed_documents"]) == 0


# =========================================================================
# 4. BM25 LEXICAL RETRIEVAL
# =========================================================================
class TestBM25LexicalRetrieval:
    def setUp(self):
        self.docs = [
            KeywordDocument(
                chunk_id="c1",
                text="Vayvora provides sub-350ms turnaround latency for voice calls.",
                source="specs.md",
                category="services",
                section="Latency",
            ),
            KeywordDocument(
                chunk_id="c2",
                text="EduSaaS offers tuition fee of five thousand rupees per track.",
                source="pricing.md",
                category="pricing",
                section="Tuition Fee",
            ),
            KeywordDocument(
                chunk_id="c3",
                text="High availability cloud infrastructure with multi-region failover.",
                source="sla.md",
                category="support",
                section="Availability",
            ),
        ]
        self.retriever = BM25Retriever(self.docs)

    def test_bm25_matches_keyword_terms(self):
        self.setUp()
        results = self.retriever.search("turnaround latency voice", top_k=2)
        assert len(results) > 0
        assert results[0].chunk_id == "c1"
        assert results[0].metadata["retrieval_method"] == "bm25"
        assert results[0].score > 0.0

    def test_bm25_empty_query(self):
        self.setUp()
        assert self.retriever.search("") == []
        assert self.retriever.search("   ") == []


# =========================================================================
# 5. RECIPROCAL RANK FUSION (RRF)
# =========================================================================
class TestReciprocalRankFusionSubsystem:
    def test_rrf_scoring_and_ranking(self):
        fusion = ReciprocalRankFusion(k=60)
        res1 = RetrievalResult(chunk_id="chunk-1", text="SLA guarantee", score=0.92, metadata={"cosine_score": 0.92})
        res2 = RetrievalResult(chunk_id="chunk-2", text="Pricing rates", score=0.85, metadata={"cosine_score": 0.85})

        vector_results = [res1, res2]
        keyword_results = [res2, res1]

        fused = fusion.fuse(vector_results, keyword_results, top_k=2)
        assert len(fused) == 2
        for r in fused:
            assert r.metadata["retrieval_method"] == "hybrid_rrf"
            assert r.score > 0.0

    def test_rrf_top_k_truncation(self):
        fusion = ReciprocalRankFusion(k=60)
        res1 = RetrievalResult(chunk_id="c1", text="A", score=0.9)
        res2 = RetrievalResult(chunk_id="c2", text="B", score=0.8)
        fused = fusion.fuse([res1, res2], [res1], top_k=1)
        assert len(fused) == 1
        assert fused[0].chunk_id == "c1"


# =========================================================================
# 6. RERANKER
# =========================================================================
class TestRerankerSubsystem:
    def test_fast_hybrid_reranking_score_calibration(self):
        reranker = Reranker()
        candidates = [
            RetrievalResult(chunk_id="r1", text="High relevance", score=0.032, metadata={"cosine_score": 0.89}),
            RetrievalResult(chunk_id="r2", text="Lower relevance", score=0.016, metadata={"cosine_score": 0.65}),
        ]
        reranked = reranker.rerank("query text", candidates, top_k=2)
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "r1"
        assert reranked[0].score == 0.89
        assert reranked[0].metadata["retrieval_method"] == "fast_hybrid_scored"

    def test_reranker_empty_inputs(self):
        reranker = Reranker()
        assert reranker.rerank("", [RetrievalResult(chunk_id="x", text="y", score=1.0)]) == []
        assert reranker.rerank("query", []) == []


# =========================================================================
# 7. RAG VECTOR STORE
# =========================================================================
class TestRAGVectorStoreSubsystem:
    def test_dimension_validation(self):
        store = RAGVectorStore()
        with pytest.raises(ValueError, match="Dimension mismatch"):
            store._normalize_vector([0.1] * 10)

    def test_zero_vector_rejection(self):
        store = RAGVectorStore()
        with pytest.raises(ValueError, match="zero vector"):
            store._normalize_vector([0.0] * 384)

    def test_add_and_search_in_memory(self):
        store = RAGVectorStore()
        dim = 384
        vec1 = [1.0] + [0.0] * (dim - 1)
        vec2 = [0.0, 1.0] + [0.0] * (dim - 2)

        c1 = DocumentChunk(chunk_id="t1", document_id="doc1", tenant_id="tenant_a", text="Apple")
        c2 = DocumentChunk(chunk_id="t2", document_id="doc2", tenant_id="tenant_b", text="Banana")

        inserted = store.add_chunks([c1, c2], [vec1, vec2])
        assert inserted == 2

        res = store.search(query_vector=vec1, top_k=1, tenant_id="tenant_a")
        assert len(res) == 1
        assert res[0].chunk_id == "t1"
        assert pytest.approx(res[0].score, rel=1e-3) == 1.0


# =========================================================================
# 8. DETERMINISTIC FAST-PATHS
# =========================================================================
class TestFastPathSubsystem:
    def test_rules_presence(self):
        assert len(FAST_PATH_RULES) >= 5

    def test_office_location_rule(self):
        res = get_deterministic_fast_path("Where is your office located?")
        assert res is not None
        assert res["topic"] == "office_headquarters"
        assert "Bangalore" in res["response"]
        assert res["bypassed_llm"] is True
        assert res["latency_ms"] < 20.0

    def test_courses_and_pricing_rules(self):
        res_course = get_deterministic_fast_path("What projects do you offer?")
        assert res_course is not None
        assert "Skyline Towers" in res_course["response"]
        assert "Green Valley" in res_course["response"]
        assert "Urban Heights" in res_course["response"]

        res_price = get_deterministic_fast_path("What is the price of Skyline Towers?")
        assert res_price is not None
        assert "₹85 Lakhs" in res_price["response"]

    def test_unmatched_queries(self):
        assert get_deterministic_fast_path("") is None
        assert get_deterministic_fast_path("What is the weather in Paris?") is None


# =========================================================================
# 9. PROPERTY RECOMMENDATIONS
# =========================================================================
class TestCourseRecommendationSubsystem:
    def test_ai_plus_fullstack_path(self):
        res = PropertyRecommendationService.get_recommendations({
            "property_type": "villa",
            "location": "Sarjapur Road",
        })
        assert "Green Valley Villas" in res["recommendedProperty"]
        assert "4 BHK" in res["configurations"][0]
        assert res["matchScore"] >= 90

    def test_cloud_devops_path(self):
        res = PropertyRecommendationService.get_recommendations({
            "interest": ["ready-to-move"],
            "location": "Electronic City",
            "budget": "50 Lakhs",
        })
        assert "Urban Heights" in res["recommendedProperty"]
        assert "Electronic City" in res["location"]
        assert res["matchScore"] >= 85


# =========================================================================
# 10. REAL QUERIES ON CURRENT KNOWLEDGE BASE (LIVE HYBRID RAG)
# =========================================================================
class TestRealQueriesHybridRAG:
    @pytest.fixture
    async def live_provider(self):
        from src.rag.ingestion.pipeline import KnowledgeIngestionPipeline
        provider = GroundedKnowledgeProvider(in_memory=True)
        pipeline = KnowledgeIngestionPipeline(knowledge_provider=provider)
        await pipeline.ingest_domain(DomainType.INFRA_BUILDER)
        yield provider
        await provider.store.close()

    @pytest.mark.asyncio
    async def test_vayvora_ai_solutions_query(self, live_provider):
        query = RAGQuery(
            domain=DomainType.INFRA_BUILDER,
            query_text="What projects does Infra Builder offer?",
            top_k=3,
            relevance_threshold=0.30,
        )
        res = await live_provider.retrieve_grounded_context(query)
        assert res.knowledge_available is True
        assert len(res.chunks) > 0
        all_content = " ".join(c.content for c in res.chunks).lower()
        assert "skyline towers" in all_content or "green valley" in all_content or "urban heights" in all_content
        for c in res.chunks:
            assert c.domain == DomainType.INFRA_BUILDER

    @pytest.mark.asyncio
    async def test_vayvora_location_query(self, live_provider):
        query = RAGQuery(
            domain=DomainType.INFRA_BUILDER,
            query_text="Where is Infra Builder located?",
            top_k=3,
            relevance_threshold=0.30,
        )
        res = await live_provider.retrieve_grounded_context(query)
        assert res.knowledge_available is True
        assert len(res.chunks) > 0
        all_content = " ".join(c.content for c in res.chunks).lower()
        assert "bangalore" in all_content

    @pytest.mark.asyncio
    async def test_edusaas_courses_query(self, live_provider):
        query = RAGQuery(
            domain=DomainType.INFRA_BUILDER,
            query_text="Tell me about Skyline Towers in Whitefield",
            top_k=3,
            relevance_threshold=0.30,
        )
        res = await live_provider.retrieve_grounded_context(query)
        assert res.knowledge_available is True
        assert len(res.chunks) > 0
        all_content = " ".join(c.content for c in res.chunks).lower()
        assert "whitefield" in all_content or "skyline" in all_content
        for c in res.chunks:
            assert c.domain == DomainType.INFRA_BUILDER

    @pytest.mark.asyncio
    async def test_edusaas_ai_fee_query(self, live_provider):
        query = RAGQuery(
            domain=DomainType.INFRA_BUILDER,
            query_text="What is the price of Green Valley Villas?",
            top_k=3,
            relevance_threshold=0.30,
        )
        res = await live_provider.retrieve_grounded_context(query)
        assert res.knowledge_available is True
        assert len(res.chunks) > 0
        all_content = " ".join(c.content for c in res.chunks).lower()
        assert "2.5" in all_content or "crore" in all_content or "villa" in all_content


# =========================================================================
# 11. RETRIEVER SERVICE WITH QUERY EXPANSION
# =========================================================================
class TestRetrieverService:
    def test_query_expansion_adds_domain_cues(self):
        expanded = RAGRetriever._expand_query("tell me about the apartments")
        assert "Infra Builder" in expanded

        already_has = RAGRetriever._expand_query("infra builder whitefield project")
        assert already_has == "infra builder whitefield project"

    @pytest.mark.asyncio
    async def test_retriever_search_method(self):
        retriever = RAGRetriever()
        res = await retriever.search(
            query="What projects does Infra Builder offer?",
            tenant_id="infra_builder",
            top_k=3,
        )
        assert isinstance(res, RAGResponse)
        assert res.query == "What projects does Infra Builder offer?"
        assert res.has_relevant_context is True
        assert len(res.results) > 0
        assert "infra_builder" in res.metadata["tenant_id"]
