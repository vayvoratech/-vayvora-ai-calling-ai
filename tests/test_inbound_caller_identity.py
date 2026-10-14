"""Tests for Inbound Caller Identity Rules and Identity Collection for Infra Builder.

Verifies:
1. For an inbound call, the only caller identity information guaranteed to be available
   at initialization is the phone number.
2. Silent fabrication, invention, or assignment of name, email, company, job title,
   location, or personal details is strictly forbidden.
3. Prohibited placeholders ("Default Caller", "caller@mail", "unknown@example.com",
   "John Doe", etc.) are rejected and never assigned.
4. Phone number received from telephony is not treated as proof of name or email.
5. Trusted caller profile lookup by phone number cleanly retrieves verified profile
   attributes when registered.
6. When an operation requires recipient phone and it is unknown, the agent naturally asks for it:
   "What is your WhatsApp number?"
7. When an operation requires name and it is unknown, the agent asks:
   "And may I have your name?"
8. Information that is not required for the immediate task is NOT requested.
9. When the caller provides information, it updates the caller profile and is used
   for the current conversation.
"""

import pytest
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.llm import MockLLMProvider
from src.core.types import ConversationStage, DomainType
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile
from src.tools.mcp_client import MockToolProvider


@pytest.fixture
def state_manager() -> ConversationStateManager:
    return ConversationStateManager()


@pytest.fixture
def tool_provider() -> MockToolProvider:
    return MockToolProvider()


class TestInboundCallerIdentityRules:
    """Verifies that inbound calls adhere strictly to caller identity constraints."""

    def test_inbound_call_only_guarantees_phone_number(
        self, state_manager: ConversationStateManager
    ):
        """At inbound call initialization, only phone number is guaranteed; all other personal attributes are None."""
        state = state_manager.create_inbound_state(
            call_id="call-inbound-phone-only",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        assert state.caller.phone == "+919845123456"
        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.company is None
        assert state.caller.job_title is None
        assert state.caller.location is None
        assert state.caller.caller_type == "unknown"
        assert state.caller.is_known() is False
        assert state.caller.has_contact_info() is True

    def test_inbound_call_rejects_placeholder_values(
        self, state_manager: ConversationStateManager
    ):
        """Placeholder values like 'Default Caller', 'caller@mail', 'unknown@example.com', 'John Doe' must be sanitized to None."""
        placeholders = [
            ("Default Caller", "caller@mail"),
            ("John Doe", "unknown@example.com"),
            ("Jane Doe", "caller@mail"),
        ]

        for idx, (dummy_name, dummy_email) in enumerate(placeholders):
            state = state_manager.create_inbound_state(
                call_id=f"call-inbound-placeholder-{idx}",
                caller_phone="+919845123456",
                caller_name=dummy_name,
                caller_email=dummy_email,
                caller_company="Acme Corp",
                domain=DomainType.INFRA_BUILDER,
            )
            assert state.caller.name is None, f"Failed for dummy_name: {dummy_name}"
            assert state.caller.email is None, f"Failed for dummy_email: {dummy_email}"
            assert state.caller.phone == "+919845123456"

    def test_update_slot_rejects_prohibited_placeholders(
        self, state_manager: ConversationStateManager
    ):
        """Updating slots during conversation must reject placeholder strings from polluting caller profile."""
        state = state_manager.create_inbound_state(
            call_id="call-inbound-slot-sanitizer",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        state.update_slot("name", "Default Caller")
        state.update_slot("email", "unknown@example.com")
        state.update_slot("caller_name", "John Doe")
        state.update_slot("caller_email", "caller@mail")

        assert state.caller.name is None
        assert state.caller.email is None
        assert "name" not in state.extracted_slots
        assert "email" not in state.extracted_slots

    def test_phone_number_not_treated_as_proof_of_name_or_email(
        self, state_manager: ConversationStateManager
    ):
        """Phone number from telephony does NOT imply or fabricate name or email."""
        state = state_manager.create_inbound_state(
            call_id="call-tel-01",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )
        assert state.caller.phone == "+919845123456"
        assert state.caller.name is None
        assert state.caller.email is None

    def test_trusted_caller_profile_lookup_by_phone(
        self, state_manager: ConversationStateManager
    ):
        """When a trusted customer profile is pre-registered, an inbound call with that phone number loads verified data."""
        trusted = CallerProfile(
            name="Elena Rostova",
            phone="+919845123456",
            email="elena.rostova@techcorp.io",
            company="TechCorp Solutions",
            caller_type="homebuyer",
        )
        state_manager.register_trusted_profile("+919845123456", trusted)

        state = state_manager.create_inbound_state(
            call_id="call-trusted-01",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        assert state.caller.name == "Elena Rostova"
        assert state.caller.email == "elena.rostova@techcorp.io"
        assert state.caller.company == "TechCorp Solutions"
        assert state.caller.is_known() is True

    def test_unknown_phone_does_not_load_trusted_profile(
        self, state_manager: ConversationStateManager
    ):
        """Unregistered phone number starts with unknown profile."""
        state = state_manager.create_inbound_state(
            call_id="call-unregistered-01",
            caller_phone="+919998887777",
            domain=DomainType.INFRA_BUILDER,
        )
        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.phone == "+919998887777"
        assert state.caller.is_known() is False


class TestIdentityCollection:
    """Verifies minimal identity collection behavior during conversation."""

    @pytest.mark.asyncio
    async def test_phone_required_and_unknown_prompts_specifically_for_phone(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When caller asks for WhatsApp brochure but phone is unknown, agent asks for WhatsApp number."""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.INFRA_BUILDER,
                detected_intent="whatsapp_information_request",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="send_message",
                    arguments={"project_name": "Skyline Towers"},
                ),
                user_facing_response="I will send the brochure.",
            )
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-req-phone",
            caller_phone=None,
            domain=DomainType.INFRA_BUILDER,
        )

        turn_result = await engine.process_user_turn(
            state, "Send me the project details on WhatsApp."
        )

        assert "whatsapp number" in turn_result.response_text.lower()
        assert state.pending_question is not None
        assert state.last_action is None

    @pytest.mark.asyncio
    async def test_name_required_and_unknown_prompts_for_name(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When an action requires caller name and it is unknown: 'And may I have your name?'"""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.INFRA_BUILDER,
                detected_intent="human_followup_request",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="create_hr_followup",
                    arguments={"notes": "Caller requested sales advisor callback"},
                ),
                user_facing_response="I'll schedule an advisor callback.",
            )
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-req-name",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        turn_result = await engine.process_user_turn(
            state, "Can someone from your property sales team call me back?"
        )

        assert "And may I have your name?" in turn_result.response_text
        assert state.pending_question == "And may I have your name?"
        assert state.caller.name is None
        assert state.last_action is None

    @pytest.mark.asyncio
    async def test_unrequired_information_is_not_requested(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When answering general property questions, the agent does NOT ask for name or email."""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.INFRA_BUILDER,
                detected_intent="property_search",
                proposed_stage=ConversationStage.INFORMATION,
                action_proposed=False,
                user_facing_response="We offer luxury apartments at Skyline Towers and private villas at Green Valley.",
            )
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-no-unneeded-info",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        turn_result = await engine.process_user_turn(
            state, "What projects do you have available?"
        )

        assert "name" not in turn_result.response_text.lower()
        assert "email" not in turn_result.response_text.lower()
        assert state.pending_question is None

    @pytest.mark.asyncio
    async def test_caller_providing_information_updates_profile_and_completes_action(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When caller provides missing phone on follow-up turn, caller profile is updated and WhatsApp message is dispatched."""
        dec1 = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_information_request",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sure. What is your WhatsApp number?",
            action_proposed=True,
            proposed_action=ProposedAction(tool_name="send_message", arguments={}),
        )
        dec2 = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_information_request",
            extracted_slots={"whatsapp_consent": True},
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sending details to your WhatsApp.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"recipient": "919845123456", "project_name": "Skyline Towers", "whatsapp_consent": True},
            ),
        )

        mock_llm = MockLLMProvider(canned_decisions=[dec1, dec2])
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-multiturn-collection",
            caller_phone=None,
            domain=DomainType.INFRA_BUILDER,
        )
        state.update_slot("preferred_project", "Skyline Towers")

        # Turn 1
        res1 = await engine.process_user_turn(state, "Send me the project brochure on WhatsApp.")
        assert "whatsapp number" in res1.response_text.lower()
        assert state.caller.phone is None

        # Turn 2
        res2 = await engine.process_user_turn(state, "My number is 9845123456")
        assert state.caller.phone == "919845123456"
        assert state.last_action == "send_message"
        assert state.last_tool_result.success is True
