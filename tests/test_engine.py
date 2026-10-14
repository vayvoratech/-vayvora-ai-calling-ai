"""Unit and integration tests for ConversationEngine orchestration with Infra Builder."""

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
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_search",
            extracted_slots={"preferred_project": "Skyline Towers", "configuration": "3 BHK"},
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Skyline Towers in Whitefield offers luxury 3 BHK apartments starting at 1.45 Crore.",
            knowledge_required=True,
            knowledge_query="Skyline Towers 3 BHK price amenities Whitefield",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-1", "+919876543210", domain=DomainType.INFRA_BUILDER)

        result: EngineTurnResult = await engine.process_user_turn(
            state, "Tell me about 3 BHK apartments at Skyline Towers."
        )

        assert result.response_text == "Skyline Towers in Whitefield offers luxury 3 BHK apartments starting at 1.45 Crore."
        assert state.current_intent == "property_search"
        assert state.extracted_slots["preferred_project"] == "Skyline Towers"
        assert state.extracted_slots["configuration"] == "3 BHK"
        assert state.stage == ConversationStage.INFORMATION
        assert state.conversation_active is True
        assert len(state.history) == 2
        assert state.history[0].role == TurnRole.CALLER
        assert state.history[1].role == TurnRole.AGENT

    @pytest.mark.asyncio
    async def test_intent_preemption_over_pending_question(self):
        """User asks a new question while agent had a pending question."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-2", "+919876543210", domain=DomainType.INFRA_BUILDER)
        state.set_intent("budget_discussion")
        state.set_pending_question("What budget range do you have in mind for your new home?")

        # Caller ignores pending question and asks for location
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="location_query",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Skyline Towers is located in Whitefield, Bangalore.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        result = await engine.process_user_turn(state, "Wait, where is Skyline Towers located?")

        # Preemption check: pending question cleared and new intent takes priority
        assert state.current_domain == DomainType.INFRA_BUILDER
        assert state.current_intent == "location_query"
        assert state.pending_question is None
        assert "budget_discussion" in state.intent_history

    @pytest.mark.asyncio
    async def test_intent_switching_preserves_context(self):
        """Switching from property inquiry to site visit keeps transcript and slots intact."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-3", "+919876543210", domain=DomainType.INFRA_BUILDER)
        state.update_slot("name", "Sameer Khan")
        state.update_slot("preferred_project", "Green Valley Villas")

        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="site_visit_request",
            extracted_slots={"configuration": "4 BHK"},
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I can arrange a site visit to Green Valley Villas for you. Which day suits you best?",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        await engine.process_user_turn(state, "Can I schedule a visit to see the 4 BHK villa?")

        assert state.current_domain == DomainType.INFRA_BUILDER
        assert state.caller.name == "Sameer Khan"
        assert state.extracted_slots["preferred_project"] == "Green Valley Villas"
        assert state.extracted_slots["configuration"] == "4 BHK"

    @pytest.mark.asyncio
    async def test_slot_sanitization(self):
        """Invalid or hallucinated slot names are filtered out."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_search",
            extracted_slots={
                "preferred_project": "Urban Heights",  # Valid
                "invented_slot_not_in_schema": "some_value",  # Invalid
            },
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Urban Heights is located in Electronic City.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-4", "+919876543210", domain=DomainType.INFRA_BUILDER)

        await engine.process_user_turn(state, "Tell me about Urban Heights.")
        assert "preferred_project" in state.extracted_slots
        assert "invented_slot_not_in_schema" not in state.extracted_slots

    @pytest.mark.asyncio
    async def test_business_status_does_not_terminate_conversation(self):
        """When business status advances to interested, conversation stays open."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="site_visit_request",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I would be happy to check available site visit slots for you.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-5", "+919876543210", domain=DomainType.INFRA_BUILDER)
        state.update_business_status("interested")

        result = await engine.process_user_turn(state, "I am definitely interested in visiting the property.")

        assert state.business_status == "interested"
        assert state.conversation_active is True
        assert result.conversation_active is True

    @pytest.mark.asyncio
    async def test_no_does_not_terminate_call(self):
        """Negative responses like 'no' must NOT close the call."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="refusal",
            proposed_stage=ConversationStage.OBJECTION_HANDLING,
            user_facing_response="No problem at all. We also have villa projects if you prefer more open space.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-6", "+919876543210", domain=DomainType.INFRA_BUILDER)

        result = await engine.process_user_turn(state, "No, I am not looking for apartments.")
        assert result.conversation_active is True
        assert result.termination_occurred is False
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_explicit_farewell_terminates_call(self):
        """Explicit farewell closes call immediately without corrupting state."""
        mock_llm = MockLLMProvider()
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-7", "+919876543210", domain=DomainType.INFRA_BUILDER)

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
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_information_request",
            extracted_slots={"whatsapp_consent": True},
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I can send the project brochure to your WhatsApp.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"recipient": "+919876543210", "message": "Infra Builder brochure for Skyline Towers"},
                rationale="Caller requested brochure via WhatsApp",
            ),
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        engine = ConversationEngine(llm_provider=mock_llm)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-eng-8", "+919876543210", domain=DomainType.INFRA_BUILDER)

        result = await engine.process_user_turn(state, "Please send me the brochure on WhatsApp.")

        # Action is recorded as pending in state
        assert result.action_proposed is True
        assert result.proposed_action.tool_name == "send_message"
        assert state.pending_action == "send_message"
        # Crucial check: tool execution result is NOT fabricated as success
        assert state.last_tool_result is None
        # Call remains active!
        assert result.conversation_active is True
        assert state.conversation_active is True
