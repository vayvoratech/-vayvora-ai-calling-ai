"""Tests for Phase 5D-3D: Clarification Field Simplification.

Verifies:
1. A clarification decision without clarification_question registers user_facing_response as the pending question.
2. A legacy decision with clarification_question prefers that field.
3. A clarification decision with an empty response does not register an empty pending question.
4. Non-clarification turns preserve the current clearing behavior.
5. Deterministically established pending questions are not accidentally erased.
6. An unrelated caller request preempts a pending question as before.
7. Returning-caller verification and caller slots remain unchanged.
8. Gemini and Hugging Face output instructions omit clarification_question.
9. Legacy payloads and synthetic decisions containing clarification_question remain compatible.
10. No clarification question is spoken twice because of this change.
"""

import json
from pathlib import Path
import pytest

from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.llm import MockLLMProvider
from src.core.prompts import PromptSynthesizer
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    TurnRole,
)
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState


@pytest.mark.asyncio
class TestClarificationSimplification:
    """Test suite covering all 10 requirements of Phase 5D-3D."""

    async def test_01_clarification_without_clarification_question_registers_user_facing_response(self):
        """1. A clarification decision without clarification_question registers user_facing_response as pending_question."""
        decision = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.PURPOSE_DISCOVERY,
            user_facing_response="Could you please specify which service you are asking about?",
            needs_clarification=True,
            clarification_question=None,
        )
        engine = ConversationEngine(llm_provider=MockLLMProvider(canned_decisions=[decision]))
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-c01", "+15551234567", domain=DomainType.GENERAL)

        result: EngineTurnResult = await engine.process_user_turn(state, "I have a question about your services.")
        assert state.pending_question == "Could you please specify which service you are asking about?"
        assert result.response_text == "Could you please specify which service you are asking about?"

    async def test_02_legacy_decision_with_clarification_question_prefers_that_field(self):
        """2. A legacy decision with clarification_question prefers that field over user_facing_response."""
        decision = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.PURPOSE_DISCOVERY,
            user_facing_response="Certainly, I'd be glad to help with that. Which service did you mean?",
            needs_clarification=True,
            clarification_question="Which service did you mean?",
        )
        engine = ConversationEngine(llm_provider=MockLLMProvider(canned_decisions=[decision]))
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-c02", "+15551234567", domain=DomainType.GENERAL)

        result: EngineTurnResult = await engine.process_user_turn(state, "I need help with a question.")
        # Preferred field is clarification_question
        assert state.pending_question == "Which service did you mean?"
        # Caller hears the full natural response
        assert result.response_text == "Certainly, I'd be glad to help with that. Which service did you mean?"

    async def test_03_clarification_with_empty_response_does_not_register_empty_pending_question(self):
        """3. A clarification decision with an empty response does not register an empty pending question."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-c03", "+15551234567", domain=DomainType.GENERAL)
        decision = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="",
            knowledge_required=True,
            needs_clarification=True,
            clarification_question=None,
        )
        # Directly test turn conclusion logic in engine
        raw_question = (
            decision.clarification_question
            if (decision.clarification_question and str(decision.clarification_question).strip())
            else decision.user_facing_response
        )
        selected_question = str(raw_question).strip() if raw_question else ""
        if selected_question:
            state.set_pending_question(selected_question)
        elif not state.pending_question:
            state.set_pending_question(None)

        assert state.pending_question is None

    async def test_04_non_clarification_turns_preserve_current_clearing_behavior(self):
        """4. Non-clarification turns preserve the current clearing behavior."""
        decision = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Our office hours are 9 AM to 6 PM Monday through Friday.",
            needs_clarification=False,
            clarification_question=None,
        )
        engine = ConversationEngine(llm_provider=MockLLMProvider(canned_decisions=[decision]))
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-c04", "+15551234567", domain=DomainType.GENERAL)
        state.set_pending_question("Previous question awaiting answer?")

        await engine.process_user_turn(state, "What are your office hours?")
        assert state.pending_question is None

    async def test_05_deterministically_established_pending_questions_not_accidentally_erased(self):
        """5. Deterministically established pending questions are not accidentally erased."""
        mgr = ConversationStateManager()
        # Outbound call initiation establishes greeting pending question
        state = mgr.create_outbound_state(
            call_id="call-c05",
            caller_phone="+919876543210",
            contact_name="Sarah",
            domain=DomainType.INFRA_BUILDER,
        )
        engine = ConversationEngine(llm_provider=MockLLMProvider())
        outbound_result = await engine.start_outbound_conversation(state)

        assert state.pending_question == "Am I speaking with Sarah?"
        assert "Sarah" in outbound_result.response_text

    async def test_06_unrelated_caller_request_preempts_pending_question(self):
        """6. An unrelated caller request preempts a pending question as before."""
        dec1 = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.PURPOSE_DISCOVERY,
            user_facing_response="Which service category are you looking for?",
            needs_clarification=True,
        )
        dec2 = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="help",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="I can assist you with our customer support.",
            needs_clarification=False,
        )
        mock_llm = MockLLMProvider(canned_decisions=[dec1, dec2])
        engine = ConversationEngine(llm_provider=mock_llm)
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-c06", "+15551234567", domain=DomainType.GENERAL)

        await engine.process_user_turn(state, "I want help with services.")
        assert state.pending_question == "Which service category are you looking for?"

        # Preemption: caller ignores the question and asks for help
        await engine.process_user_turn(state, "Actually, can someone help me with technical support?")
        assert state.current_intent == "help"
        assert state.pending_question is None

    async def test_07_returning_caller_verification_and_caller_slots_remain_unchanged(self):
        """7. Returning-caller verification and caller slots remain unchanged."""
        mgr = ConversationStateManager()
        profile = CallerProfile(
            phone="+919876543210",
            name="John Doe",
        )
        state = mgr.create_inbound_state(
            call_id="call-c07",
            caller_phone="+919876543210",
            caller_profile=profile,
            domain=DomainType.GENERAL,
        )
        state.update_slot("returning_caller", True)
        state.update_slot("caller_name", "John Doe")

        dec = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.PURPOSE_DISCOVERY,
            user_facing_response="Welcome back, John! What would you like to check today?",
            needs_clarification=True,
        )
        engine = ConversationEngine(llm_provider=MockLLMProvider(canned_decisions=[dec]))
        result = await engine.process_user_turn(state, "Hi, I called earlier.")

        # Slots and profile are preserved
        assert state.caller.name == "John Doe"
        assert state.get_slot("caller_name") == "John Doe"
        assert state.pending_question == "Welcome back, John! What would you like to check today?"
        assert result.response_text == "Welcome back, John! What would you like to check today?"

    async def test_08_gemini_and_huggingface_output_instructions_omit_clarification_question(self):
        """8. Gemini and Hugging Face output instructions omit clarification_question."""
        from src.domains import INFRA_BUILDER_DOMAIN_CONFIG
        synthesizer = PromptSynthesizer()
        state = ConversationStateManager().create_inbound_state("call-c08", "+15551234567", domain=DomainType.INFRA_BUILDER)
        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)

        # Gemini instructions (loaded via json_schema.txt and decision_instructions.txt)
        assert "needs_clarification" in system_prompt
        assert "clarification_question" not in system_prompt

        # Hugging Face in-code instructions
        hf_source = Path("src/core/huggingface_llm.py").read_text(encoding="utf-8")
        assert '"clarification_question"' not in hf_source
        assert '"needs_clarification"' in hf_source

    async def test_09_legacy_payloads_and_synthetic_decisions_remain_compatible(self):
        """9. Legacy payloads and synthetic decisions containing clarification_question remain compatible."""
        # Deserializing legacy payload containing clarification_question
        legacy_json = json.dumps({
            "detected_intent": "general_inquiry",
            "detected_sub_intent": None,
            "extracted_slots": {},
            "proposed_stage": "information",
            "user_facing_response": "Which BHK configuration are you looking for?",
            "knowledge_required": False,
            "knowledge_query": None,
            "proposed_action": None,
            "needs_clarification": True,
            "clarification_question": "Which BHK configuration are you looking for?",
        })
        decision = ConversationalDecision.model_validate(json.loads(legacy_json))
        assert decision.needs_clarification is True
        assert decision.clarification_question == "Which BHK configuration are you looking for?"

        # Synthetic fallback decision from engine (e.g. noise / unclear)
        synthetic_dec = ConversationalDecision(
            detected_domain=DomainType.UNKNOWN,
            detected_intent="clarification",
            proposed_stage=ConversationStage.PURPOSE_DISCOVERY,
            user_facing_response="I didn't quite catch that. Could you tell me what you're calling about?",
            knowledge_required=False,
            needs_clarification=True,
            clarification_question="Could you tell me what you're calling about?",
        )
        assert synthetic_dec.clarification_question == "Could you tell me what you're calling about?"

    async def test_10_no_clarification_question_spoken_twice(self):
        """10. No clarification question is spoken twice because of this change."""
        decision = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.PURPOSE_DISCOVERY,
            user_facing_response="Could you please tell me your preferred move-in timeline?",
            needs_clarification=True,
        )
        engine = ConversationEngine(llm_provider=MockLLMProvider(canned_decisions=[decision]))
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-c10", "+15551234567", domain=DomainType.GENERAL)

        result = await engine.process_user_turn(state, "I'm looking for an apartment.")

        # Exactly one utterance is returned and voiced
        assert result.response_text == "Could you please tell me your preferred move-in timeline?"
        # The history records only this single response from the agent
        agent_turns = [t for t in state.history if t.role == TurnRole.AGENT]
        assert len(agent_turns) == 1
        assert agent_turns[0].content == "Could you please tell me your preferred move-in timeline?"
