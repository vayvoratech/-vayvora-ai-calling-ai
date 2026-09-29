"""Tests for Inbound Caller Identity Rules and Identity Collection.

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
6. When an operation requires email and it is unknown, the agent naturally asks for it:
   "Sure. What email address should I send them to?"
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
            caller_phone="+14155559876",
            domain=DomainType.EDUSAAS,
        )

        assert state.caller.phone == "+14155559876"
        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.company is None
        assert state.caller.job_title is None
        assert state.caller.location is None
        assert state.caller.caller_type == "unknown"
        assert state.caller.is_known() is False
        assert state.caller.has_contact_info() is True  # has phone

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
                caller_phone="+15550001111",
                caller_name=dummy_name,
                caller_email=dummy_email,
                caller_company="Acme Corp",
            )
            assert state.caller.name is None, f"Failed for dummy_name: {dummy_name}"
            assert state.caller.email is None, f"Failed for dummy_email: {dummy_email}"
            assert state.caller.phone == "+15550001111"

    def test_update_slot_rejects_prohibited_placeholders(
        self, state_manager: ConversationStateManager
    ):
        """Updating slots during conversation must reject placeholder strings from polluting caller profile."""
        state = state_manager.create_inbound_state(
            call_id="call-inbound-slot-sanitizer",
            caller_phone="+15552223333",
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
            caller_phone="+18005550199",
            domain=DomainType.VAYVORA,
        )
        assert state.caller.phone == "+18005550199"
        assert state.caller.name is None
        assert state.caller.email is None

    def test_trusted_caller_profile_lookup_by_phone(
        self, state_manager: ConversationStateManager
    ):
        """When a trusted customer profile is pre-registered, an inbound call with that phone number loads verified data."""
        # 1. Pre-register a verified trusted profile
        trusted = CallerProfile(
            name="Elena Rostova",
            phone="+14155554321",
            email="elena.rostova@techcorp.io",
            company="TechCorp Solutions",
            caller_type="corporate_client",
        )
        state_manager.register_trusted_profile("+14155554321", trusted)

        # 2. Inbound call from registered phone
        state = state_manager.create_inbound_state(
            call_id="call-trusted-01",
            caller_phone="+14155554321",
            domain=DomainType.VAYVORA,
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
            caller_phone="+19998887777",
            domain=DomainType.VAYVORA,
        )
        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.phone == "+19998887777"
        assert state.caller.is_known() is False


class TestIdentityCollection:
    """Verifies minimal identity collection behavior during conversation."""

    @pytest.mark.asyncio
    async def test_email_required_and_unknown_prompts_specifically_for_email(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When caller asks for email materials but email is unknown: 'Sure. What email address should I send them to?'"""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="request_course_syllabus",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="send_email",
                    arguments={"subject": "AI Course Syllabus"},
                ),
                user_facing_response="I will email the syllabus.",
            )
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-req-email",
            caller_phone="+15558881234",
            domain=DomainType.EDUSAAS,
        )

        turn_result = await engine.process_user_turn(
            state, "Send me the course details by email."
        )

        # Agent must prompt specifically for email address without executing tool
        assert "What email address should I send them to?" in turn_result.response_text
        assert state.pending_question is not None
        assert state.caller.email is None
        assert state.last_action is None

    @pytest.mark.asyncio
    async def test_name_required_and_unknown_prompts_for_name(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When an action requires caller name and it is unknown: 'And may I have your name?'"""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="request_advisor_callback",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="create_hr_followup",
                    arguments={"notes": "Caller requested technical consultation callback"},
                ),
                user_facing_response="I'll schedule a consultant callback.",
            )
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-req-name",
            caller_phone="+15557774321",
            domain=DomainType.VAYVORA,
        )

        turn_result = await engine.process_user_turn(
            state, "Can someone from your engineering team call me back?"
        )

        # Agent prompts for caller's name
        assert "And may I have your name?" in turn_result.response_text
        assert state.pending_question == "And may I have your name?"
        assert state.caller.name is None
        assert state.last_action is None

    @pytest.mark.asyncio
    async def test_unrequired_information_is_not_requested(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When answering general or course questions, the agent does NOT ask for name or email."""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="course_inquiry",
                proposed_stage=ConversationStage.INFORMATION,
                action_proposed=False,
                user_facing_response="We offer Data Science and Voice AI engineering programs with hands-on labs.",
            )
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-no-unneeded-info",
            caller_phone="+15551239999",
            domain=DomainType.EDUSAAS,
        )

        turn_result = await engine.process_user_turn(
            state, "What courses do you offer?"
        )

        assert "name" not in turn_result.response_text.lower()
        assert "email" not in turn_result.response_text.lower()
        assert state.pending_question is None

    @pytest.mark.asyncio
    async def test_caller_providing_information_updates_profile_and_completes_action(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When caller provides missing email on follow-up turn, caller profile is updated and email is dispatched."""
        # Turn 1: Propose email action without email
        dec1 = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="request_syllabus",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sure. What email address should I send them to?",
            action_proposed=True,
            proposed_action=ProposedAction(tool_name="send_email", arguments={}),
        )
        # Turn 2: Caller provides email
        dec2 = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="request_syllabus",
            extracted_slots={"email": "rahul.sharma@gmail.com"},
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="Sending course details to rahul.sharma@gmail.com.",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "rahul.sharma@gmail.com", "subject": "Course Syllabus"},
            ),
        )

        mock_llm = MockLLMProvider(
            canned_decisions=[dec1, dec2],
            canned_responses=["I have sent the syllabus to rahul.sharma@gmail.com."],
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-multiturn-collection",
            caller_phone="+15553337777",
            domain=DomainType.EDUSAAS,
        )

        # Turn 1
        res1 = await engine.process_user_turn(state, "Send me the syllabus by email.")
        assert "What email address should I send them to?" in res1.response_text
        assert state.caller.email is None

        # Turn 2
        res2 = await engine.process_user_turn(state, "My email is rahul.sharma@gmail.com")
        assert state.caller.email == "rahul.sharma@gmail.com"
        assert state.last_action == "send_email"
        assert state.last_tool_result.success is True
