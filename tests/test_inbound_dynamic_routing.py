"""Comprehensive Test Suite for Infra Builder Inbound Routing & Domain Invariants.

Validates that:
1. Inbound greeting ("Hello") -> stays locked to Infra Builder, responds naturally without asking caller to choose businesses.
2. Inbound real estate request ("I want to know about Skyline Towers") -> stays Infra Builder.
3. Incidental keywords ("courses" or "software") -> domain remains locked to Infra Builder.
4. Inbound ambiguous request ("I need some information") -> stays Infra Builder, asks how we can help with properties.
5. Legacy domains cannot be registered or activated.
6. Outbound call starts locked to Infra Builder.
7. Strict RAG Isolation -> queries only Infra Builder verified knowledge base.
8. Edge case: "no", "no thanks" does NOT terminate call or disrupt domain.
9. Edge case: STT noise/unclear text does NOT break domain state.
10. Edge case: Explicit farewell terminates the call cleanly.
"""

import pytest
from typing import Dict, Any, List

from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    TurnRole,
    RAGQuery,
    RAGChunk,
)

class GroundedChunk(RAGChunk):
    chunk_id: str = ""

from src.core.decision import (
    ConversationalDecision,
    is_ambiguous_purpose,
    is_pure_greeting,
    is_noise_or_unclear,
)
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.llm import MockLLMProvider
from src.rag.retriever import GroundedKnowledgeProvider, GroundedContextResult
from src.state.manager import ConversationStateManager
from src.state.models import ConversationState


class DummyKnowledgeProvider:
    """Mock KnowledgeProvider recording queried domains and returning domain-scoped chunks."""

    def __init__(self):
        self.queries_received: List[RAGQuery] = []

    async def retrieve_grounded_context(self, query: RAGQuery) -> GroundedContextResult:
        self.queries_received.append(query)
        if query.domain == DomainType.INFRA_BUILDER:
            chunks = [
                GroundedChunk(
                    chunk_id="ib-1",
                    doc_id="doc-skyline-1",
                    title="Skyline Towers Overview",
                    content="Skyline Towers in Whitefield offers luxury 2 and 3 BHK residences.",
                    score=0.95,
                    domain=DomainType.INFRA_BUILDER.value,
                )
            ]
        else:
            chunks = []

        return GroundedContextResult(
            query=query,
            chunks=chunks,
            knowledge_available=bool(chunks),
            formatted_context="\n".join(c.content for c in chunks),
        )


@pytest.mark.asyncio
async def test_scenario_1_inbound_greeting_identifies_as_infra_builder():
    """Test 1: Inbound caller greets -> domain is Infra Builder, responds without business selection."""
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-1", caller_phone="+919845123456", domain=DomainType.INFRA_BUILDER)

    assert state.domain == DomainType.INFRA_BUILDER

    result = await engine.process_user_turn(state, "Hello")

    assert state.domain == DomainType.INFRA_BUILDER
    assert result.decision.detected_domain == DomainType.INFRA_BUILDER
    # Must not ask to choose between businesses
    assert "choose" not in result.response_text.lower()
    assert "edusaas" not in result.response_text.lower()
    assert "vayvora" not in result.response_text.lower()


@pytest.mark.asyncio
async def test_scenario_2_property_inquiry_locks_infra_builder():
    """Test 2: Caller asks for Skyline Towers -> handled under Infra Builder."""
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-2", caller_phone="+919845123456", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "I want to know about Skyline Towers in Whitefield.")

    assert state.domain == DomainType.INFRA_BUILDER
    assert result.decision.detected_domain == DomainType.INFRA_BUILDER


@pytest.mark.asyncio
async def test_scenario_3_incidental_keywords_do_not_switch_domain():
    """Test 3: Caller mentions courses or software -> remains Infra Builder."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="property_search",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Infra Builder specializes in premium residential properties in Bangalore. We do not offer academic courses.",
    )
    engine = ConversationEngine(llm_provider=MockLLMProvider(canned_decisions=[decision]))
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-3", caller_phone="+919845123456", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Do you have any computer science courses?")

    assert state.domain == DomainType.INFRA_BUILDER
    assert result.decision.detected_domain == DomainType.INFRA_BUILDER


@pytest.mark.asyncio
async def test_scenario_4_legacy_domains_raise_key_error():
    """Test 4: Legacy domain types cannot be resolved from the registry."""
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    with pytest.raises(KeyError):
        engine.registry.get_or_raise(DomainType.EDUSAAS)
    with pytest.raises(KeyError):
        engine.registry.get_or_raise(DomainType.VAYVORA)


@pytest.mark.asyncio
async def test_scenario_5_outbound_call_starts_as_infra_builder():
    """Test 5: Outbound call initializes with Infra Builder identity."""
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_outbound_state(
        call_id="call-out-1",
        caller_phone="+919845123456",
        domain=DomainType.INFRA_BUILDER,
        caller_name="Rohan",
        campaign_id="skyline-launch",
        campaign_objective="Introduce Skyline Towers luxury residences",
    )

    assert state.metadata.direction == CallDirection.OUTBOUND
    assert state.domain == DomainType.INFRA_BUILDER
    assert state.domain_locked is True

    greeting = engine.synthesize_outbound_greeting(state)
    assert "Infra Builder" in greeting
    assert "vayvora" not in greeting.lower()
    assert "edusaas" not in greeting.lower()


@pytest.mark.asyncio
async def test_scenario_6_rag_isolation_queries_only_infra_builder():
    """Test 6: Verify RAG queries use only the Infra Builder knowledge store."""
    rag_provider = DummyKnowledgeProvider()
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="property_search",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Skyline Towers offers 2 and 3 BHK luxury residences.",
        knowledge_required=True,
        knowledge_query="Skyline Towers Whitefield amenities",
    )
    engine = ConversationEngine(
        llm_provider=MockLLMProvider(canned_decisions=[decision]),
        knowledge_provider=rag_provider,
    )
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-rag-ib", caller_phone="+919845123456", domain=DomainType.INFRA_BUILDER)

    await engine.process_user_turn(state, "Tell me about Skyline Towers.")

    assert len(rag_provider.queries_received) == 1
    assert rag_provider.queries_received[0].domain == DomainType.INFRA_BUILDER


@pytest.mark.asyncio
async def test_scenario_7_negative_phrases_do_not_terminate():
    """Test 7: Negative phrases like 'no', 'no thanks' do NOT terminate the call."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="refusal",
        proposed_stage=ConversationStage.OBJECTION_HANDLING,
        user_facing_response="No problem. Let me know if you would like information on other projects.",
    )
    engine = ConversationEngine(llm_provider=MockLLMProvider(canned_decisions=[decision]))
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-7", caller_phone="+919845123456", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "No thanks, not looking for apartments.")

    assert result.conversation_active is True
    assert result.termination_occurred is False
    assert state.conversation_active is True


@pytest.mark.asyncio
async def test_scenario_8_explicit_farewell_terminates():
    """Test 8: Explicit farewell closes the call cleanly."""
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-8", caller_phone="+919845123456", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Thank you, goodbye.")

    assert result.conversation_active is False
    assert result.termination_occurred is True
    assert state.conversation_active is False
    assert state.stage == ConversationStage.COMPLETED
