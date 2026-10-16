"""Focused regression tests for RAG query expansion in src/rag/services/retriever.py.

Verifies:
1. A query asking for the location of Pristine City retains the project name and does not acquire Bangalore-related terms.
2. A query asking about JB Pride amenities preserves the specific question.
3. A query about Nature Valley and Choutuppal does not acquire unrelated locations.
4. A query containing an explicit location retains that location without introducing contradictory locations.
5. A query that requires no expansion remains unchanged.
6. Existing supported domain-specific behavior remains compatible.
7. Corrected queries retrieve the expected project documents through the existing retriever implementation.
"""

import pytest
import src.core.types  # Ensures core types are initialized before RAG imports
from src.core.types import DomainType, RAGChunk, RAGQuery
from src.rag.retriever import GroundedKnowledgeProvider
from src.rag.services.retriever import RAGRetriever


BANGALORE_TERMS = {
    "bangalore",
    "bengaluru",
    "whitefield",
    "sarjapur",
    "electronic city",
    "urban heights",
    "skyline towers",
    "green valley",
}


# =============================================================================
# 1. PRISTINE CITY LOCATION QUERY
# =============================================================================
class TestPristineCityQueryExpansion:
    """Case 1: Location of Pristine City retains project name and has no Bangalore terms."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is the location of Pristine City?",
            "Where is Pristine City located?",
            "How do I reach Pristine City project?",
            "Tell me about the location of Pristine City",
        ],
    )
    def test_pristine_city_location_retains_project_and_no_bangalore_terms(self, query: str):
        expanded = RAGRetriever._expand_query(query, domain=DomainType.INFRA_BUILDER)
        expanded_lower = expanded.lower()

        # Project name and location inquiry must be preserved
        assert "pristine city" in expanded_lower
        # No Bangalore-related terms must be injected
        for term in BANGALORE_TERMS:
            assert term not in expanded_lower, f"Found contaminated term '{term}' in '{expanded}'"


# =============================================================================
# 2. JB PRIDE AMENITIES QUERY
# =============================================================================
class TestJBPrideAmenitiesQueryExpansion:
    """Case 2: JB Pride amenities preserves the specific question."""

    @pytest.mark.parametrize(
        "query",
        [
            "What amenities does JB Pride have?",
            "Tell me about the amenities at JB Pride project",
            "Does JB Pride have a clubhouse and swimming pool?",
        ],
    )
    def test_jb_pride_amenities_preserves_specific_question(self, query: str):
        expanded = RAGRetriever._expand_query(query, domain=DomainType.INFRA_BUILDER)
        expanded_lower = expanded.lower()

        # Specific question details must be preserved
        assert "jb pride" in expanded_lower
        assert any(cue in expanded_lower for cue in ("amenities", "clubhouse", "swimming pool"))
        # No Bangalore terms or unrelated locations
        for term in BANGALORE_TERMS:
            assert term not in expanded_lower, f"Found contaminated term '{term}' in '{expanded}'"


# =============================================================================
# 3. NATURE VALLEY AND CHOUTUPPAL QUERY
# =============================================================================
class TestNatureValleyQueryExpansion:
    """Case 3: Nature Valley and Choutuppal does not acquire unrelated locations."""

    @pytest.mark.parametrize(
        "query",
        [
            "Tell me about Nature Valley in Choutuppal",
            "What is the plot price at Nature Valley project in Choutuppal?",
            "Where is Nature Valley located in Choutuppal?",
        ],
    )
    def test_nature_valley_choutuppal_does_not_acquire_unrelated_locations(self, query: str):
        expanded = RAGRetriever._expand_query(query, domain=DomainType.INFRA_BUILDER)
        expanded_lower = expanded.lower()

        # Project and location must be preserved
        assert "nature valley" in expanded_lower
        assert "choutuppal" in expanded_lower
        # No unrelated locations
        for term in BANGALORE_TERMS:
            assert term not in expanded_lower, f"Found contaminated term '{term}' in '{expanded}'"


# =============================================================================
# 4. EXPLICIT LOCATION QUERIES
# =============================================================================
class TestExplicitLocationQueryExpansion:
    """Case 4: Query containing an explicit location retains it without contradictory locations."""

    @pytest.mark.parametrize(
        "query,expected_loc",
        [
            ("Do you have plots near Ibrahimpatnam?", "ibrahimpatnam"),
            ("Show me approved layouts in Choutuppal", "choutuppal"),
            ("What properties are near TSPA Junction?", "tspa"),
            ("Are there gated community plots in Vikarabad?", "vikarabad"),
        ],
    )
    def test_explicit_location_retained_without_contradictory_locations(
        self, query: str, expected_loc: str
    ):
        expanded = RAGRetriever._expand_query(query, domain=DomainType.INFRA_BUILDER)
        expanded_lower = expanded.lower()

        # Original explicit location must be preserved
        assert expected_loc in expanded_lower
        # Contradictory locations must not be introduced
        for term in BANGALORE_TERMS:
            assert term not in expanded_lower, f"Found contaminated term '{term}' in '{expanded}'"


# =============================================================================
# 5. QUERIES REQUIRING NO EXPANSION
# =============================================================================
class TestNoExpansionQueries:
    """Case 5: A query that requires no expansion remains unchanged."""

    @pytest.mark.parametrize(
        "query",
        [
            "What projects does Infra Builder offer?",
            "infra builder whitefield project",
            "Infra Builder contact number",
            "What is the refund policy?",
            "hello",
            "good morning",
            "",
            "   ",
        ],
    )
    def test_query_requiring_no_expansion_remains_unchanged(self, query: str):
        expanded = RAGRetriever._expand_query(query, domain=DomainType.INFRA_BUILDER)
        assert expanded == query


# =============================================================================
# 6. DOMAIN ISOLATION & COMPATIBILITY
# =============================================================================
class TestDomainIsolationAndCompatibility:
    """Case 6: Existing supported domain-specific behavior remains compatible."""

    def test_non_infra_builder_domains_remain_unchanged(self):
        edusaas_query = "Tell me about the AI/ML course and fees"
        assert RAGRetriever._expand_query(edusaas_query, domain=DomainType.EDUSAAS) == edusaas_query
        assert RAGRetriever._expand_query(edusaas_query, domain="edusaas") == edusaas_query

        vayvora_query = "What voice agent integrations are available?"
        assert RAGRetriever._expand_query(vayvora_query, domain=DomainType.VAYVORA) == vayvora_query
        assert RAGRetriever._expand_query(vayvora_query, domain="vayvora") == vayvora_query

    def test_company_overview_query_expansion(self):
        query = "what is infra builder"
        expanded = RAGRetriever._expand_query(query, domain=DomainType.INFRA_BUILDER)
        assert "what is infra builder" in expanded.lower()
        assert "company overview" in expanded.lower()
        assert "real estate developer" in expanded.lower()
        for term in BANGALORE_TERMS:
            assert term not in expanded.lower()

    def test_generic_real_estate_query_expansion(self):
        query = "tell me about the apartments"
        expanded = RAGRetriever._expand_query(query, domain=DomainType.INFRA_BUILDER)
        assert "tell me about the apartments" in expanded
        assert "Infra Builder" in expanded
        for term in BANGALORE_TERMS:
            assert term not in expanded.lower()


# =============================================================================
# 7. RETRIEVAL IMPLEMENTATION VERIFICATION
# =============================================================================
class TestRetrievalVerificationWithCorrectedQuery:
    """Verify that corrected queries retrieve the expected project documents."""

    @pytest.mark.asyncio
    async def test_retriever_search_retrieves_pristine_city(self):
        retriever = RAGRetriever()
        res = await retriever.search(
            query="Where is Pristine City located?",
            tenant_id="infra_builder",
            top_k=3,
        )
        assert res.has_relevant_context is True
        assert len(res.results) > 0
        # Pristine City or overview catalog must be retrieved
        all_text = " ".join(r.text for r in res.results).lower()
        all_sources = " ".join(r.source for r in res.results).lower()
        assert "pristine city" in all_text or "pristine_city" in all_sources
        # Ensure enriched query does not contain Bangalore terms
        assert "bangalore" not in res.metadata["enriched_query"].lower()

    @pytest.mark.asyncio
    async def test_retriever_search_retrieves_nature_valley_choutuppal(self):
        retriever = RAGRetriever()
        res = await retriever.search(
            query="Tell me about Nature Valley in Choutuppal",
            tenant_id="infra_builder",
            top_k=3,
        )
        assert res.has_relevant_context is True
        assert len(res.results) > 0
        all_text = " ".join(r.text for r in res.results).lower()
        all_sources = " ".join(r.source for r in res.results).lower()
        assert "nature valley" in all_text or "choutuppal" in all_text or "nature_valley" in all_sources
        assert "bangalore" not in res.metadata["enriched_query"].lower()

    @pytest.mark.asyncio
    async def test_retriever_search_retrieves_jb_pride_amenities(self):
        retriever = RAGRetriever()
        res = await retriever.search(
            query="What amenities does JB Pride have?",
            tenant_id="infra_builder",
            top_k=3,
        )
        assert res.has_relevant_context is True
        assert len(res.results) > 0
        all_text = " ".join(r.text for r in res.results).lower()
        assert "jb pride" in all_text or "amenities" in all_text
        assert "bangalore" not in res.metadata["enriched_query"].lower()

    @pytest.mark.asyncio
    async def test_retriever_search_retrieves_ibrahimpatnam_projects(self):
        retriever = RAGRetriever()
        res = await retriever.search(
            query="Where is Serene City located in Ibrahimpatnam?",
            tenant_id="infra_builder",
            top_k=3,
        )
        assert res.has_relevant_context is True
        assert len(res.results) > 0
        all_text = " ".join(r.text for r in res.results).lower()
        assert "ibrahimpatnam" in all_text
        assert "serene city" in all_text
        assert "bangalore" not in res.metadata["enriched_query"].lower()
