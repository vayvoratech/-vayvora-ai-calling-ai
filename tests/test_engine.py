"""Unit and integration tests for ConversationEngine orchestration."""

import pytest
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.llm import MockLLMProvider
from src.core.types import ConversationStage, DomainType, TurnRole
from src.state.manager import ConversationStateManager


class TestConversationEngine:
    """Test turn orchestration, state progression, and safety guardrails."""

    @pytest.mark.asyncio
    async def test_valid_structured_turn(self):
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_information",
            extracted_slots={"target_course": "AI Engineering", "experience": "beginner"},
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Our AI Engineering course starts with Python fundamentals.",
            knowledge_required=True,
            knowledge_query="AI Engineering syllabus beginner",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-1", "+15551234567", domain=DomainType.EDUSAAS)

        result: EngineTurnResult = await engine.process_user_turn(
            state, "Tell me about your AI Engineering course for beginners."
        )

        assert result.response_text == "Our AI Engineering course starts with Python fundamentals."
        assert state.current_intent == "course_information"
        assert state.extracted_slots["target_course"] == "AI Engineering"
        assert state.extracted_slots["experience"] == "beginner"
        assert state.stage == ConversationStage.INFORMATION
        assert state.conversation_active is True
        assert len(state.history) == 2
        assert state.history[0].role == TurnRole.CALLER
        assert state.history[1].role == TurnRole.AGENT

    @pytest.mark.asyncio
    async def test_intent_preemption_over_pending_question(self):
        """User asks a new question while agent had a pending question."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-2", "+15551234567", domain=DomainType.EDUSAAS)
        state.set_intent("course_recommendation")
        state.set_pending_question("What degree are you currently pursuing?")

        # Caller ignores pending question and asks for office location
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="company_location",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Vayvora's engineering center is located in Bangalore.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        result = await engine.process_user_turn(state, "Wait, where is your office located?")

        # Preemption check
        assert state.current_domain == DomainType.VAYVORA
        assert state.current_intent == "company_location"
        assert state.pending_question is None  # Prior pending question cleared!
        assert "course_recommendation" in state.intent_history

    @pytest.mark.asyncio
    async def test_domain_switching_preserves_context(self):
        """Switching from EduSaaS to Vayvora keeps transcript and slots intact."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-3", "+15551234567", domain=DomainType.EDUSAAS)
        state.update_slot("student_name", "Sameer Khan")
        state.update_slot("interest", "LLMs")

        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="career_information",
            extracted_slots={"desired_role": "AI Engineer"},
            proposed_stage=ConversationStage.DISCOVERY,
            user_facing_response="Vayvora is currently hiring AI Engineers for our voice agent team.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        await engine.process_user_turn(state, "Does your parent company Vayvora hire AI engineers?")

        assert state.current_domain == DomainType.VAYVORA
        assert state.primary_domain == DomainType.EDUSAAS
        assert state.caller.name == "Sameer Khan"
        assert state.extracted_slots["interest"] == "LLMs"
        assert state.extracted_slots["desired_role"] == "AI Engineer"

    @pytest.mark.asyncio
    async def test_slot_sanitization(self):
        """Invalid or hallucinated slot names are filtered out."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_information",
            extracted_slots={
                "target_course": "AI Bootcamp",  # Valid
                "invented_slot_not_in_schema": "some_value",  # Invalid
            },
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="We have an AI Bootcamp.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-4", "+15551234567", domain=DomainType.EDUSAAS)

        await engine.process_user_turn(state, "I want the AI Bootcamp.")
        assert "target_course" in state.extracted_slots
        assert "invented_slot_not_in_schema" not in state.extracted_slots

    @pytest.mark.asyncio
    async def test_business_status_does_not_terminate_conversation(self):
        """When business status advances to interested, conversation stays open."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="enrollment",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I can certainly help you with enrollment.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-5", "+15551234567", domain=DomainType.EDUSAAS)
        state.update_business_status("interested")

        result = await engine.process_user_turn(state, "I am definitely interested in enrolling.")

        assert state.business_status == "interested"
        assert state.conversation_active is True
        assert result.conversation_active is True

    @pytest.mark.asyncio
    async def test_no_does_not_terminate_call(self):
        """Negative responses like 'no' must NOT close the call."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="objection",
            proposed_stage=ConversationStage.OBJECTION_HANDLING,
            user_facing_response="No problem at all. We also offer introductory modules.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-6", "+15551234567", domain=DomainType.EDUSAAS)

        result = await engine.process_user_turn(state, "No, I do not have coding experience.")
        assert result.conversation_active is True
        assert result.termination_occurred is False
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_explicit_farewell_terminates_call(self):
        """Explicit farewell closes call immediately without corrupting state."""
        mock_llm = MockLLMProvider()
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-7", "+15551234567", domain=DomainType.EDUSAAS)

        result = await engine.process_user_turn(state, "That's all, thank you goodbye.")
        assert result.conversation_active is False
        assert result.termination_occurred is True
        assert state.conversation_active is False
        assert state.stage == ConversationStage.COMPLETED
        assert "caller_explicit_farewell" in state.termination_reason

    @pytest.mark.asyncio
    async def test_prevention_of_fabricated_tool_success_claims(self):
        """Proposed actions remain pending; tool success is NOT assumed."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_information",
            extracted_slots={"email": "student@example.com"},
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I have queued the syllabus to be sent to your email.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="tools_email_send",
                arguments={"email": "student@example.com", "course": "AI Engineering"},
                rationale="Caller requested syllabus",
            ),
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-8", "+15551234567", domain=DomainType.EDUSAAS)

        result = await engine.process_user_turn(state, "Please send me the syllabus to student@example.com")

        # Action is recorded as pending in state
        assert result.action_proposed is True
        assert result.proposed_action.tool_name == "tools_email_send"
        assert state.pending_action == "tools_email_send"
        # Crucial check: tool execution result is NOT fabricated as success
        assert state.last_tool_result is None
        # Call remains active!
        assert result.conversation_active is True
        assert state.conversation_active is True
