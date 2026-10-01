"""Comprehensive Test Suite for Inbound Dynamic Domain Routing & Locking.

Validates the 10 core scenarios plus voice edge cases:
1. Inbound greeting ("Hello") -> domain remains unknown, unlocked.
2. Inbound Vayvora request ("AI solutions") -> locks to vayvora.
3. Locked Vayvora + other domain keyword ("courses") -> domain remains locked to vayvora.
4. Locked Vayvora + explicit correction ("Actually, I meant EduSaaS...") -> switches and locks to edusaas.
5. Fresh inbound EduSaaS request ("education platform") -> locks to edusaas.
6. Fresh inbound ambiguous request ("I need some information") -> domain remains unknown, asks clarification.
7. Fresh inbound software request ("software development services") -> locks to vayvora.
8. Outbound Vayvora call -> starts locked to vayvora, untouched by dynamic routing.
9. Outbound EduSaaS call -> starts locked to edusaas, untouched by dynamic routing.
10. Strict RAG Isolation -> UNKNOWN skips RAG; locked domains query only their authorized knowledge store.
11. Edge case: "no", "no thanks" does NOT switch domain or terminate.
12. Edge case: STT noise/unclear text does NOT lock domain.
13. Edge case: Inbound API/WebSocket cannot force locked domain over dynamic routing.
14. Edge case: Explicit farewell terminates the call cleanly.
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
    classify_inbound_domain,
    detect_explicit_domain_correction,
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
        if query.domain == DomainType.VAYVORA:
            chunks = [
                GroundedChunk(
                    chunk_id="v-1",
                    doc_id="doc-vayvora-1",
                    title="Vayvora AI Engineering",
                    content="Vayvora provides enterprise AI solutions and custom LLM agents.",
                    score=0.95,
                    domain=DomainType.VAYVORA.value,
                )
            ]
        elif query.domain == DomainType.EDUSAAS:
            chunks = [
                GroundedChunk(
                    chunk_id="e-1",
                    doc_id="doc-edusaas-1",
                    title="EduSaaS Courses",
                    content="EduSaaS offers comprehensive full-stack and data science courses.",
                    score=0.92,
                    domain=DomainType.EDUSAAS.value,
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
async def test_scenario_1_greeting_keeps_unknown_unlocked():
    """Test 1: Caller -> 'Hello'.
    Expected: domain = unknown, domain_locked = false.
    Agent asks how to help without assuming a business domain.
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-1", caller_phone="+15550000001")

    assert state.domain == DomainType.UNKNOWN
    assert state.domain_locked is False

    result = await engine.process_user_turn(state, "Hello")

    assert state.domain == DomainType.UNKNOWN
    assert state.domain_locked is False
    assert result.decision.detected_domain == DomainType.UNKNOWN
    assert "help" in result.response_text.lower()


@pytest.mark.asyncio
async def test_scenario_2_ai_solutions_locks_vayvora():
    """Test 2: Caller -> 'I want to know about your AI solutions.'
    Expected: domain = vayvora, domain_locked = true.
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-2", caller_phone="+15550000002")

    result = await engine.process_user_turn(state, "I want to know about your AI solutions.")

    assert state.domain == DomainType.VAYVORA
    assert state.domain_locked is True
    assert result.decision.detected_domain == DomainType.VAYVORA


@pytest.mark.asyncio
async def test_scenario_3_courses_keyword_does_not_switch_locked_vayvora():
    """Test 3: Caller -> 'Tell me about your courses.' after Vayvora was locked.
    Expected: domain remains vayvora, domain_locked = true.
    """
    decision = ConversationalDecision(
        detected_domain=DomainType.EDUSAAS,  # LLM erroneously suggests edusaas based on keyword
        detected_intent="course_information",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="At Vayvora, we do not provide academic courses.",
    )
    engine = ConversationEngine(llm_provider=MockLLMProvider(canned_decisions=[decision]))
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-3", caller_phone="+15550000003")

    # First establish and lock Vayvora
    state.lock_domain(DomainType.VAYVORA)
    assert state.domain == DomainType.VAYVORA
    assert state.domain_locked is True

    # Next turn: caller mentions "courses"
    result = await engine.process_user_turn(state, "Tell me about your courses.")

    # Domain lock MUST override LLM's detected_domain
    assert state.domain == DomainType.VAYVORA
    assert state.domain_locked is True
    assert result.decision.detected_domain == DomainType.VAYVORA


@pytest.mark.asyncio
async def test_scenario_4_explicit_correction_switches_and_locks():
    """Test 4: Caller -> 'Actually, I meant EduSaaS. I'm calling about the education platform.'
    Expected: domain = edusaas, domain_locked = true.
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-4", caller_phone="+15550000004")

    # Start locked to Vayvora
    state.lock_domain(DomainType.VAYVORA)
    assert state.domain == DomainType.VAYVORA
    assert state.domain_locked is True

    # Caller explicitly corrects domain
    result = await engine.process_user_turn(
        state, "Actually, I meant EduSaaS. I'm calling about the education platform."
    )

    assert state.domain == DomainType.EDUSAAS
    assert state.domain_locked is True
    assert result.decision.detected_domain == DomainType.EDUSAAS


@pytest.mark.asyncio
async def test_scenario_5_fresh_inbound_education_platform_locks_edusaas():
    """Test 5: Fresh inbound caller -> 'I want information about your education platform.'
    Expected: domain = edusaas, domain_locked = true.
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-5", caller_phone="+15550000005")

    result = await engine.process_user_turn(state, "I want information about your education platform.")

    assert state.domain == DomainType.EDUSAAS
    assert state.domain_locked is True
    assert result.decision.detected_domain == DomainType.EDUSAAS


@pytest.mark.asyncio
async def test_scenario_6_fresh_inbound_ambiguous_purpose_clarification():
    """Test 6: Fresh inbound caller -> 'I need some information.'
    Expected: domain remains unknown until clarification.
    Response asks: 'Sure. Are you calling about our software and AI solutions, or our education services?'
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-6", caller_phone="+15550000006")

    result = await engine.process_user_turn(state, "I need some information.")

    assert state.domain == DomainType.UNKNOWN
    assert state.domain_locked is False
    assert result.decision.detected_domain == DomainType.UNKNOWN
    assert "software and ai solutions" in result.response_text.lower()
    assert "education services" in result.response_text.lower()


@pytest.mark.asyncio
async def test_scenario_7_fresh_inbound_software_services_locks_vayvora():
    """Test 7: Fresh inbound caller -> 'I need help with your software development services.'
    Expected: domain = vayvora, domain_locked = true.
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-7", caller_phone="+15550000007")

    result = await engine.process_user_turn(
        state, "I need help with your software development services."
    )

    assert state.domain == DomainType.VAYVORA
    assert state.domain_locked is True
    assert result.decision.detected_domain == DomainType.VAYVORA


@pytest.mark.asyncio
async def test_scenario_8_outbound_vayvora_remains_unchanged():
    """Test 8: Outbound Vayvora call.
    Expected: starts with domain = vayvora, domain_locked = true, outbound behavior untouched.
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_outbound_state(
        call_id="call-outbound-vayvora",
        caller_phone="+15559998888",
        domain=DomainType.VAYVORA,
        caller_name="Alice Smith",
        company="Acme Corp",
        campaign_objective="exploring AI solutions",
    )

    assert state.metadata.direction == CallDirection.OUTBOUND
    assert state.domain == DomainType.VAYVORA
    assert state.domain_locked is True

    # Outbound greeting check
    greeting = engine.synthesize_outbound_greeting(state)
    assert "Vayvora" in greeting
    assert "Alice" in greeting

    # Process first turn
    result = await engine.process_user_turn(state, "Yes, tell me more.")
    assert state.domain == DomainType.VAYVORA
    assert state.domain_locked is True


@pytest.mark.asyncio
async def test_scenario_9_outbound_edusaas_remains_unchanged():
    """Test 9: Outbound EduSaaS call.
    Expected: starts with domain = edusaas, domain_locked = true, outbound behavior untouched.
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_outbound_state(
        call_id="call-outbound-edusaas",
        caller_phone="+15557776666",
        domain=DomainType.EDUSAAS,
        caller_name="Bob Kumar",
        campaign_objective="interest in Python and Data Science courses",
    )

    assert state.metadata.direction == CallDirection.OUTBOUND
    assert state.domain == DomainType.EDUSAAS
    assert state.domain_locked is True

    # Outbound greeting check
    greeting = engine.synthesize_outbound_greeting(state)
    assert "EduSaaS" in greeting
    assert "Bob" in greeting

    # Process first turn
    result = await engine.process_user_turn(state, "Sure, I have a few minutes.")
    assert state.domain == DomainType.EDUSAAS
    assert state.domain_locked is True


@pytest.mark.asyncio
async def test_scenario_10_rag_isolation_enforces_locked_domain():
    """Test 10: Verify RAG retrieval uses the locked domain and cannot retrieve the wrong company's knowledge.
    - When domain is UNKNOWN: RAG is completely skipped.
    - When locked to VAYVORA: only Vayvora knowledge retrieved.
    - When locked to EDUSAAS: only EduSaaS knowledge retrieved.
    """
    rag_provider = DummyKnowledgeProvider()
    engine = ConversationEngine(
        llm_provider=MockLLMProvider(),
        knowledge_provider=rag_provider,
    )
    mgr = ConversationStateManager()

    # Case A: Domain is UNKNOWN -> RAG must NOT be performed
    state_unknown = mgr.create_inbound_state("call-rag-unknown", caller_phone="+15551112222")
    assert state_unknown.domain == DomainType.UNKNOWN
    await engine.process_user_turn(state_unknown, "I need some information.")
    assert len(rag_provider.queries_received) == 0

    # Case B: Domain locked to VAYVORA -> Only Vayvora knowledge is retrieved
    state_vayvora = mgr.create_inbound_state("call-rag-vayvora", caller_phone="+15552223333")
    state_vayvora.lock_domain(DomainType.VAYVORA)
    decision_vayvora = ConversationalDecision(
        detected_domain=DomainType.VAYVORA,
        detected_intent="services",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Here is what we do at Vayvora.",
        knowledge_required=True,
        knowledge_query="custom AI agents and LLM engineering",
    )
    engine_v = ConversationEngine(
        llm_provider=MockLLMProvider(canned_decisions=[decision_vayvora]),
        knowledge_provider=rag_provider,
    )
    await engine_v.process_user_turn(state_vayvora, "Tell me about your custom AI agents.")
    assert len(rag_provider.queries_received) == 1
    query_v = rag_provider.queries_received[-1]
    assert query_v.domain == DomainType.VAYVORA
    assert query_v.domain != DomainType.EDUSAAS

    # Case C: Domain locked to EDUSAAS -> Only EduSaaS knowledge is retrieved
    state_edusaas = mgr.create_inbound_state("call-rag-edusaas", caller_phone="+15553334444")
    state_edusaas.lock_domain(DomainType.EDUSAAS)
    decision_edusaas = ConversationalDecision(
        detected_domain=DomainType.EDUSAAS,
        detected_intent="course_information",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Here are our course offerings.",
        knowledge_required=True,
        knowledge_query="full-stack and data science syllabus",
    )
    engine_e = ConversationEngine(
        llm_provider=MockLLMProvider(canned_decisions=[decision_edusaas]),
        knowledge_provider=rag_provider,
    )
    await engine_e.process_user_turn(state_edusaas, "What courses do you teach?")
    assert len(rag_provider.queries_received) == 2
    query_e = rag_provider.queries_received[-1]
    assert query_e.domain == DomainType.EDUSAAS
    assert query_e.domain != DomainType.VAYVORA


@pytest.mark.asyncio
async def test_scenario_11_negative_phrases_do_not_trigger_domain_correction():
    """Test 11: Negative phrases like 'no', 'no thanks', 'not really' do NOT switch domain or terminate."""
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-11", caller_phone="+15554445555")
    state.lock_domain(DomainType.VAYVORA)

    # Negative phrase
    result = await engine.process_user_turn(state, "No thanks, not right now.")
    assert state.domain == DomainType.VAYVORA
    assert state.domain_locked is True
    assert result.conversation_active is True
    assert result.termination_occurred is False


@pytest.mark.asyncio
async def test_scenario_12_stt_noise_asks_clarification_without_locking():
    """Test 12: STT produces noise / unclear text.
    Expected: domain remains unknown, domain_locked remains false, agent asks for clarification.
    """
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-12", caller_phone="+15556667777")

    result = await engine.process_user_turn(state, "[noise]")
    assert state.domain == DomainType.UNKNOWN
    assert state.domain_locked is False
    assert "didn't quite catch that" in result.response_text.lower()


@pytest.mark.asyncio
async def test_scenario_13_frontend_api_inbound_domain_param_not_authoritative():
    """Test 13: Inbound session created through service or API initializes with UNKNOWN and unlocked."""
    from src.ui.service import WorkbenchService
    from src.state.manager import ConversationStateManager

    state_mgr = ConversationStateManager()
    workbench = WorkbenchService(
        state_manager=state_mgr,
        engine=ConversationEngine(llm_provider=MockLLMProvider()),
    )

    # Client passes domain=vayvora in request, but inbound session must NOT be locked to it
    state = workbench.create_inbound_session(
        call_id="call-test-api-inbound",
        caller_phone="+15558889999",
        domain=DomainType.VAYVORA,
    )
    assert state.domain == DomainType.UNKNOWN
    assert state.domain_locked is False


@pytest.mark.asyncio
async def test_scenario_14_call_termination_preserved():
    """Test 14: Caller explicit farewell closes the call cleanly."""
    engine = ConversationEngine(llm_provider=MockLLMProvider())
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-test-14", caller_phone="+15550001111")
    state.lock_domain(DomainType.VAYVORA)

    result = await engine.process_user_turn(state, "Thank you, goodbye!")
    assert result.conversation_active is False
    assert result.termination_occurred is True
    assert state.conversation_active is False
