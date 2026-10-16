"""Focused test suite for Phase 2: Conversational Response & Follow-Up Prompt Optimization.

Covers:
1. A factual answer followed by at most one relevant question.
2. No follow-up after a greeting or simple acknowledgment.
3. No follow-up after goodbye or call termination.
4. No repeated question when caller already supplied information.
5. Changed topic takes precedence over previous topic.
6. Insufficient RAG evidence does not invent facts.
7. Pricing remains strictly grounded in verified evidence.
8. Completed action described only using verified results.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import pytest

import src.core.types  # Enforce initialization order
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.interfaces import KnowledgeProvider, LLMProvider, ToolProvider
from src.core.prompts import PromptSynthesizer
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    RAGChunk,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
    TurnRole,
)
from src.domains import INFRA_BUILDER_DOMAIN_CONFIG
from src.prompts import load_prompt, render_prompt
from src.rag.retriever import GroundedContextResult
from src.state.manager import ConversationStateManager
from src.state.models import ConversationState


# =============================================================================
# Mock Providers for Deterministic Turn Testing
# =============================================================================

class MockLLMProvider(LLMProvider):
    """Deterministic LLM Provider that returns canned decisions or text."""

    def __init__(
        self,
        canned_decisions: Optional[List[ConversationalDecision]] = None,
        canned_text_responses: Optional[List[str]] = None,
    ) -> None:
        self.canned_decisions = list(canned_decisions or [])
        self.canned_text_responses = list(canned_text_responses or [])
        self.decision_prompts: List[str] = []
        self.response_prompts: List[Dict[str, str]] = []

    async def generate_decision(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> ConversationalDecision:
        self.decision_prompts.append(prompt)
        if self.canned_decisions:
            return self.canned_decisions.pop(0)
        return ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="general_question",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Default mock decision response.",
            knowledge_required=False,
        )

    async def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        self.response_prompts.append({"prompt": prompt, "system_instruction": system_instruction or ""})
        if self.canned_text_responses:
            return self.canned_text_responses.pop(0)
        return "Default mock text response."

    async def stream_response(self, prompt: str, system_instruction: Optional[str] = None):
        res = await self.generate_response(prompt, system_instruction)
        yield res


class MockKnowledgeProvider(KnowledgeProvider):
    """Deterministic Knowledge Provider for RAG testing."""

    def __init__(self, chunks: Optional[List[RAGChunk]] = None) -> None:
        self.chunks = list(chunks or [])
        self.queries: List[RAGQuery] = []

    async def search(self, query: RAGQuery) -> List[RAGChunk]:
        self.queries.append(query)
        return self.chunks

    async def retrieve_grounded_context(self, query: RAGQuery) -> GroundedContextResult:
        self.queries.append(query)
        if not self.chunks:
            return GroundedContextResult(
                query=query,
                chunks=[],
                knowledge_available=False,
                formatted_context="No verified knowledge found passing relevance threshold.",
            )

        context_lines = [
            f'<verified_reference_data domain="{query.domain.value}" untrusted="true">',
        ]
        for idx, c in enumerate(self.chunks, 1):
            context_lines.append(f"--- Document {idx}: {c.title} ---")
            context_lines.append(f"Source: {c.metadata.get('source_file', 'unknown')}")
            context_lines.append(f"Content:\n{c.content}\n")
        context_lines.append("</verified_reference_data>")

        return GroundedContextResult(
            query=query,
            chunks=self.chunks,
            knowledge_available=True,
            formatted_context="\n".join(context_lines),
        )

    async def index_document(self, chunk: RAGChunk) -> bool:
        self.chunks.append(chunk)
        return True


class MockToolProvider(ToolProvider):
    """Deterministic Tool Provider for action verification tests."""

    def __init__(self, return_data: Optional[Dict[str, Any]] = None, success: bool = True) -> None:
        self.return_data = return_data or {"status": "success"}
        self.success = success
        self.executed_tools: List[ToolCallRequest] = []

    async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
        self.executed_tools.append(request)
        return ToolExecutionResult(
            tool_name=request.tool_name,
            success=self.success,
            data=self.return_data,
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "find_available_slots", "description": "Check availability"},
            {"name": "create_calendar_event", "description": "Book a site visit"},
            {"name": "send_message", "description": "Send WhatsApp message"},
        ]


# =============================================================================
# 1. FACTUAL ANSWER FOLLOWED BY AT MOST ONE RELEVANT QUESTION
# =============================================================================

def test_prompt_instructions_enforce_answer_first_and_one_followup():
    """Deterministic prompt check: Prompts explicitly mandate answering first, followed by at most one relevant question."""
    grounded_answer = load_prompt("rag/grounded_answer.txt")
    rag_system = load_prompt("rag/system.txt")
    decision_instructions = load_prompt("agent/decision_instructions.txt")
    rules = load_prompt("common/rules.txt")

    # All key prompt files must mandate Answer First
    assert "Answer first" in grounded_answer or "answer the caller's latest question directly" in grounded_answer.lower()
    assert "Answer first" in rag_system or "answer it directly" in rag_system.lower()
    assert "Answer first" in decision_instructions
    assert "Answer first" in rules

    # Prompts must mandate at most one relevant follow-up
    assert "At most one relevant follow-up" in grounded_answer or "at most one concise" in grounded_answer.lower()
    assert "At most one relevant follow-up" in rag_system or "at most one concise" in rag_system.lower()
    assert "At most one relevant follow-up" in decision_instructions
    assert "At most one relevant follow-up" in rules


@pytest.mark.asyncio
async def test_factual_answer_followed_by_one_relevant_question():
    """Verify that a factual RAG turn answers first with verified info, followed by at most one relevant follow-up."""
    chunk = RAGChunk(
        doc_id="pristine-loc-1",
        domain=DomainType.INFRA_BUILDER,
        score=0.95,
        content="Pristine City is a premium HMDA-approved plotted layout located in Choutuppal along the Hyderabad-Vijayawada Highway (NH-65).",
        metadata={"project": "Pristine City", "source_file": "knowledge_base/infra_builder/projects/pristine_city.md"},
        title="Pristine City Location",
    )
    mock_kp = MockKnowledgeProvider(chunks=[chunk])

    # LLM Step 1: Decision requires knowledge
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="location_query",
        extracted_slots={"preferred_project": "Pristine City"},
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Pristine City is in Choutuppal.",
        knowledge_required=True,
        knowledge_query="Pristine City location",
    )
    # LLM Step 2: Grounded synthesis adheres to prompt: answers first, then asks one relevant question
    grounded_synthesis_response = (
        "Pristine City is located in Choutuppal along the Hyderabad-Vijayawada Highway NH-65. "
        "Would you like directions or details on available plot sizes?"
    )

    mock_llm = MockLLMProvider(
        canned_decisions=[decision],
        canned_text_responses=[grounded_synthesis_response],
    )
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-phase2-1", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Where is Pristine City located?")

    # 1. Directly answers the location first
    assert "Pristine City is located in Choutuppal" in result.response_text
    # 2. Asks at most one question mark
    assert result.response_text.count("?") == 1
    # 3. Question appears after the answer, not before it
    ans_idx = result.response_text.find("Choutuppal")
    q_idx = result.response_text.find("?")
    assert ans_idx < q_idx
    # 4. Prompt passed to LLM contained the verified reference data and strict instructions
    assert len(mock_llm.response_prompts) == 1
    passed_prompt = mock_llm.response_prompts[0]["prompt"]
    assert "Pristine City is a premium HMDA-approved plotted layout located in Choutuppal" in passed_prompt


# =============================================================================
# 2. NO FOLLOW-UP AFTER GREETING OR SIMPLE ACKNOWLEDGMENT
# =============================================================================

def test_prompt_instructions_forbid_followup_after_greeting_or_acknowledgment():
    """Deterministic prompt check: Prompts forbid forcing follow-up questions after greetings or acknowledgments."""
    grounded_answer = load_prompt("rag/grounded_answer.txt")
    rag_system = load_prompt("rag/system.txt")
    decision_instructions = load_prompt("agent/decision_instructions.txt")
    rules = load_prompt("common/rules.txt")

    for prompt_content in [grounded_answer, rag_system, decision_instructions, rules]:
        content_lower = prompt_content.lower()
        assert "greetings" in content_lower
        assert "acknowledg" in content_lower


@pytest.mark.asyncio
async def test_no_followup_after_greeting_or_simple_acknowledgment():
    """Verify turn processing does not force follow-ups or RAG retrieval for simple acknowledgments."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="acknowledgment",
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Understood.",
        knowledge_required=False,
    )
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    mock_kp = MockKnowledgeProvider(chunks=[])

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-phase2-2", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Okay, sure.")

    # Does not invoke RAG for simple acknowledgments
    assert len(mock_kp.queries) == 0
    # Response has no forced questions
    assert "?" not in result.response_text


# =============================================================================
# 3. NO FOLLOW-UP AFTER GOODBYE / TERMINATION
# =============================================================================

def test_prompt_instructions_forbid_followup_after_goodbye():
    """Deterministic prompt check: Prompts forbid follow-up questions after goodbye or call termination."""
    grounded_answer = load_prompt("rag/grounded_answer.txt")
    rag_system = load_prompt("rag/system.txt")
    decision_instructions = load_prompt("agent/decision_instructions.txt")
    rules = load_prompt("common/rules.txt")

    for prompt_content in [grounded_answer, rag_system, decision_instructions, rules]:
        assert "goodbye" in prompt_content.lower()


@pytest.mark.asyncio
async def test_no_followup_after_goodbye():
    """Verify goodbye responses do not append follow-up questions."""
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="end_call",
        proposed_stage=ConversationStage.CLOSING,
        user_facing_response="Thank you for calling Infra Builder. Have a wonderful day!",
        knowledge_required=False,
        suggested_termination=True,
    )
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    engine = ConversationEngine(llm_provider=mock_llm)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-phase2-3", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Thanks, that is all. Goodbye!")

    # No questions asked after goodbye
    assert "?" not in result.response_text
    assert "goodbye" in result.response_text.lower() or "great day" in result.response_text.lower()
    assert result.termination_occurred is True


# =============================================================================
# 4. NO REPEATED QUESTION WHEN CALLER ALREADY SUPPLIED INFORMATION
# =============================================================================

def test_prompt_synthesizer_injects_known_slots_and_forbids_repeated_questions():
    """Deterministic prompt check: PromptSynthesizer surfaces persistent slots and prompts forbid re-asking known info."""
    synthesizer = PromptSynthesizer()
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-phase2-4", "+919876543210", domain=DomainType.INFRA_BUILDER)

    # Pre-populate known caller slots
    state.update_slot("preferred_project", "JB Pride")
    state.update_slot("budget", "60 lakhs")
    state.update_slot("phone", "+919876543210")

    user_prompt = synthesizer.build_user_prompt(state, "What amenities are available?")
    system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)

    # Known slots must appear in user prompt persistent memory
    assert "=== PERSISTENT CONVERSATION SLOTS ===" in user_prompt
    assert "- preferred_project: JB Pride" in user_prompt
    assert "- budget: 60 lakhs" in user_prompt

    # Negative constraint forbidding repeated questions is in system prompt, excluded from user prompt
    assert "=== DECISION INSTRUCTIONS ===" not in user_prompt
    assert (
        "Never ask for information the caller has already provided" in system_prompt
        or "Never ask for details" in system_prompt
    )
    assert "repeat a question whose answer is already known" in system_prompt.lower()


# =============================================================================
# 5. CHANGED TOPIC TAKES PRECEDENCE OVER PREVIOUS TOPIC
# =============================================================================

def test_changed_topic_precedence_in_prompt_synthesis():
    """Deterministic prompt check: When caller changes topic, prompt synthesizer explicitly flags pending question as ignorable."""
    synthesizer = PromptSynthesizer()
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-phase2-5", "+919876543210", domain=DomainType.INFRA_BUILDER)

    # Agent had asked for a site visit date
    state.set_pending_action("find_available_slots")
    state.set_pending_question("Which date would you prefer for the site visit?")
    state.set_stage(ConversationStage.ACTION_CONFIRMATION)

    # Caller interrupts and changes topic
    new_message = "Actually, what amenities does Nature Valley have?"
    user_prompt = synthesizer.build_user_prompt(state, new_message)
    system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)

    # 1. Latest caller message is marked as primary signal in user prompt; static instructions excluded
    assert "=== LATEST CALLER MESSAGE\n(PRIMARY SIGNAL FOR CURRENT INTENT) ===" in user_prompt
    assert f'"{new_message}"' in user_prompt
    assert "=== DECISION INSTRUCTIONS ===" not in user_prompt

    # 2. System prompt instructions instruct model to ignore pending question if caller asked something else
    assert 'Ignore it if caller asked something else' in system_prompt

    # 3. System prompt behavioral rules explicitly mandate handling new topic first
    assert (
        "If the caller changes topic or interrupts, handle the new request first" in system_prompt
        or "handle the new request first" in system_prompt.lower()
    )


# =============================================================================
# 6. INSUFFICIENT RAG EVIDENCE DOES NOT INVENT FACTS
# =============================================================================

def test_prompt_instructions_forbid_inventing_facts_on_insufficient_evidence():
    """Deterministic prompt check: Prompts forbid inventing missing facts when RAG context is insufficient."""
    grounded_answer = load_prompt("rag/grounded_answer.txt")
    rag_system = load_prompt("rag/system.txt")

    assert "If the requested fact is absent, stale, irrelevant, or unverified in the context, say that the specific information cannot currently be confirmed" in grounded_answer
    assert "Do not fill the gap with a typical or expected answer" in grounded_answer
    assert "Never fabricate or estimate" in rag_system
    assert "Do not use general model knowledge, assumptions, estimates, guesses, or inferred facts" in rag_system


@pytest.mark.asyncio
async def test_insufficient_rag_evidence_does_not_invent_facts():
    """Verify that when RAG returns insufficient results, engine returns fail-closed fallback without hallucinations."""
    # Knowledge provider has NO chunks for the query
    mock_kp = MockKnowledgeProvider(chunks=[])

    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="amenities_query",
        extracted_slots={"preferred_project": "Pristine City"},
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Pristine City has an Olympic swimming pool and helipad.",  # Hallucinated draft
        knowledge_required=True,
        knowledge_query="Pristine City helipad",
    )
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-phase2-6", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Does Pristine City have a private helipad?")

    # Initial draft with hallucinated facts was discarded
    assert "helipad" not in result.response_text.lower()
    assert "olympic swimming pool" not in result.response_text.lower()
    # Safe fallback returned
    assert "verified details regarding that at the moment" in result.response_text or "verified property details" in result.response_text


# =============================================================================
# 7. PRICING REMAINS GROUNDED IN VERIFIED EVIDENCE
# =============================================================================

def test_prompt_instructions_strict_numeric_and_pricing_grounding():
    """Deterministic prompt check: Prompts strictly forbid estimating, calculating, or inventing prices."""
    grounded_answer = load_prompt("rag/grounded_answer.txt")
    rag_system = load_prompt("rag/system.txt")
    rules = load_prompt("common/rules.txt")

    assert "Numeric facts—including prices, fees, dimensions, dates, counts, discounts, and payment amounts—must be explicitly present" in grounded_answer
    assert "Numeric information is especially strict: state a number only when that exact fact is explicitly supported" in rag_system
    assert "Numeric facts—including prices, fees, discounts, dates, dimensions, counts, percentages, possession timelines, and payment terms—must be explicitly supported by trusted knowledge" in rules


@pytest.mark.asyncio
async def test_pricing_remains_strictly_grounded_in_verified_evidence():
    """Verify that pricing responses only quote explicit verified lines and reject unsupported rates."""
    chunk = RAGChunk(
        doc_id="pristine-price-1",
        domain=DomainType.INFRA_BUILDER,
        score=0.95,
        content="Pristine City plot prices start from Rs. 14,500 per sq. yard. Commercial plots are Rs. 18,000 per sq. yard.",
        metadata={"project": "Pristine City", "source_file": "knowledge_base/infra_builder/projects/pristine_city.md"},
        title="Pristine City Pricing",
    )
    mock_kp = MockKnowledgeProvider(chunks=[chunk])

    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="price_query",
        extracted_slots={"preferred_project": "Pristine City"},
        proposed_stage=ConversationStage.INFORMATION,
        user_facing_response="Plots are very cheap.",
        knowledge_required=True,
        knowledge_query="Pristine City price",
    )
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-phase2-7", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "What is the price per square yard for Pristine City?")

    # Verified price evidence line is explicitly quoted
    assert "14,500 per sq. yard" in result.response_text
    assert "The retrieved project information for Pristine City states:" in result.response_text


# =============================================================================
# 8. COMPLETED ACTION DESCRIBED ONLY USING VERIFIED RESULTS
# =============================================================================

def test_prompt_instructions_forbid_claiming_unverified_action_success():
    """Deterministic prompt check: Prompts strictly forbid claiming action completion without verified result."""
    grounded_answer = load_prompt("rag/grounded_answer.txt")
    rag_system = load_prompt("rag/system.txt")
    decision_instructions = load_prompt("agent/decision_instructions.txt")
    rules = load_prompt("common/rules.txt")

    assert "Do not claim a site visit was booked, a WhatsApp message was sent or delivered, a representative was notified, or any other external action was completed unless a verified tool result explicitly confirms" in grounded_answer
    assert "Do not claim that a WhatsApp message was sent or delivered, a site visit was booked, a representative was notified, or any other action was completed unless the relevant verified tool result" in rag_system
    assert "Do not claim a tool action has succeeded before a successful verified result is available" in decision_instructions
    assert "Never claim success unless the tool actually executed, returned a successful result, and the result was verified" in rules


@pytest.mark.asyncio
async def test_completed_action_described_only_using_verified_results():
    """Verify that action completion confirmation quotes only verified backend results and asks one relevant next question."""
    mock_tp = MockToolProvider(
        return_data={"available_slots": ["Tomorrow at 10:00 AM", "Tomorrow at 2:00 PM"]},
        success=True,
    )
    decision = ConversationalDecision(
        detected_domain=DomainType.INFRA_BUILDER,
        detected_intent="book_site_visit",
        extracted_slots={"preferred_project": "JB Pride", "preferred_date": "tomorrow"},
        proposed_stage=ConversationStage.ACTION_CONFIRMATION,
        user_facing_response="I will check slots.",
        action_proposed=True,
        proposed_action=ProposedAction(
            tool_name="find_available_slots",
            arguments={"property_name": "JB Pride", "preferred_date": "tomorrow"},
            rationale="User requested available slots",
        ),
        knowledge_required=False,
    )
    mock_llm = MockLLMProvider(canned_decisions=[decision])
    engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tp)
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state("call-phase2-8", "+919876543210", domain=DomainType.INFRA_BUILDER)

    result = await engine.process_user_turn(state, "Can you check site visit slots for JB Pride tomorrow?")

    # 1. Action was executed
    assert len(mock_tp.executed_tools) == 1
    # 2. Response accurately reflects only verified slots returned by the tool
    assert "Tomorrow at 10:00 AM" in result.response_text
    assert "Tomorrow at 2:00 PM" in result.response_text
    # 3. Asks exactly one relevant follow-up question to choose a slot
    assert result.response_text.count("?") == 1
    assert "Which one would you prefer?" in result.response_text
    # 4. Did not claim a booking was already confirmed
    assert "confirmed" not in result.response_text.lower()
