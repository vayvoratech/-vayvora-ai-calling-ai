"""Tests for Caller Information & Calendar Slot Handling (Bug 1 & Bug 2).

Verifies:
1. Complete elimination of sample/fallback caller data (no Alice, no dummy emails/phones).
2. Unknown inbound and partial outbound caller profiles remain cleanly None/empty.
3. Canonical 'meeting_preference' slot validation across all domains (EduSaaS, Vayvora, General).
4. Date/time extraction (e.g. 'today by 12 PM', 'Tomorrow at 10 AM') into 'meeting_preference'.
5. Persistence of 'meeting_preference' across conversational turns.
6. Calendar tool execution guard automatically reusing 'meeting_preference' without repeatedly asking.
7. Slot revision when caller updates preference ('actually, make it 2 PM instead').
"""

import pytest
from src.core.decision import ConversationalDecision, DecisionValidator, ProposedAction, extract_datetime_preference
from src.core.engine import ConversationEngine
from src.core.llm import MockLLMProvider
from src.core.types import CallDirection, ConversationStage, DomainType
from src.state.manager import ConversationStateManager
from src.tools.mcp_client import MockToolProvider
from src.ui.app import CreateSessionRequest, create_session, get_service
from src.ui.service import WorkbenchService


@pytest.fixture
def state_manager() -> ConversationStateManager:
    return ConversationStateManager()


@pytest.fixture
def tool_provider() -> MockToolProvider:
    return MockToolProvider()


@pytest.fixture
def decision_validator() -> DecisionValidator:
    return DecisionValidator()


# =============================================================================
# BUG 1 TESTS: REMOVE ALL SAMPLE / FALLBACK CALLER DATA
# =============================================================================


class TestBug1RemoveSampleCallerData:
    """Verifies that no synthetic or fallback caller details are populated when missing."""

    def test_inbound_unknown_caller_leaves_all_profile_fields_none(
        self, state_manager: ConversationStateManager
    ):
        """Inbound call with no caller identity must have None for name, email, phone, company."""
        state = state_manager.create_inbound_state(
            call_id="call-inbound-anon",
            caller_phone=None,
            domain=DomainType.VAYVORA,
            caller_name=None,
            caller_email=None,
            caller_company=None,
        )

        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.phone is None
        assert state.caller.company is None
        assert state.caller.is_known() is False
        assert state.caller.has_contact_info() is False

    def test_outbound_caller_only_uses_explicit_fields(
        self, state_manager: ConversationStateManager
    ):
        """Outbound call with partial target contact data must NOT invent missing fields."""
        state = state_manager.create_outbound_state(
            call_id="call-outbound-partial",
            caller_phone="+14155552671",
            domain=DomainType.EDUSAAS,
            caller_name="Dr. John Doe",
            campaign_id="CAMP-TEST-01",
            campaign_objective="Discuss Masterclass",
            caller_email=None,
            company=None,
            known_purpose=None,
        )

        assert state.caller.name == "Dr. John Doe"
        assert state.caller.phone == "+14155552671"
        assert state.caller.email is None
        assert state.caller.company is None
        assert state.caller.known_purpose is None
        assert "Alice" not in str(state.caller.name)

    def test_api_create_session_request_defaults_are_none(self):
        """CreateSessionRequest Pydantic schema must default caller details to None."""
        req = CreateSessionRequest()
        assert req.caller_name is None
        assert req.caller_phone is None
        assert req.caller_email is None
        assert req.caller_company is None
        assert req.campaign_objective is None
        assert req.known_purpose is None

    @pytest.mark.asyncio
    async def test_api_create_session_endpoint_inbound_clean_none(self):
        """Calling create_session endpoint without caller info creates an anonymous session."""
        req = CreateSessionRequest(
            direction="inbound",
            domain="vayvora",
        )
        resp = await create_session(req)
        assert resp["session_id"] is not None
        service = get_service()
        state = service.get_session(resp["session_id"])
        assert state is not None
        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.phone is None
        assert state.caller.company is None

    @pytest.mark.asyncio
    async def test_missing_email_action_does_not_invent_sample_email(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When email is missing, the engine asks the caller for it instead of inventing a dummy."""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="request_brochure",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="send_email",
                    arguments={"subject": "Brochure"},
                ),
                user_facing_response="I will send over the brochure.",
            )
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-email-guard",
            domain=DomainType.VAYVORA,
        )

        turn_result = await engine.process_user_turn(state, "Please send me the brochure.")

        # Email guard should intercept and request email without executing tool
        assert "Could you please share your email address?" in turn_result.response_text
        assert state.caller.email is None
        assert "alice" not in turn_result.response_text.lower()
        assert "example.com" not in turn_result.response_text.lower()
        assert state.last_action is None


# =============================================================================
# BUG 2 TESTS: INBOUND DATE/TIME & CALENDAR SLOT HANDLING
# =============================================================================


class TestBug2CalendarSlotHandling:
    """Verifies canonical 'meeting_preference' extraction, persistence, and execution."""

    def test_extract_datetime_preference_helper(self):
        """Tests natural-language date/time preference extraction helper."""
        assert extract_datetime_preference("today by 12 PM") == "today by 12 PM"
        assert extract_datetime_preference("Tomorrow at 3 PM") == "Tomorrow at 3 PM"
        assert extract_datetime_preference("tomorrow afternoon") == "tomorrow afternoon"
        assert extract_datetime_preference("next Tuesday at 10 AM IST") == "next Tuesday at 10 AM IST"
        assert extract_datetime_preference("actually, make it 2 PM instead") == "2 PM"
        assert extract_datetime_preference("can we do 2:00 PM") == "2:00 PM"
        assert extract_datetime_preference("What courses do you offer?") is None
        assert extract_datetime_preference("Hello") is None

    def test_decision_validator_preserves_meeting_preference_in_all_domains(
        self, decision_validator: DecisionValidator
    ):
        """DecisionValidator must never strip meeting_preference in EduSaaS, Vayvora, or General."""
        for domain in [DomainType.EDUSAAS, DomainType.VAYVORA, DomainType.GENERAL]:
            decision = ConversationalDecision(
                detected_domain=domain,
                detected_intent="meeting_request" if domain == DomainType.VAYVORA else "course_information",
                extracted_slots={"meeting_preference": "today by 12 PM"},
                user_facing_response="Understood, 12 PM works.",
            )
            validated = decision_validator.validate_and_filter(decision)
            assert "meeting_preference" in validated.extracted_slots
            assert validated.extracted_slots["meeting_preference"] == "today by 12 PM"

    def test_decision_validator_canonicalizes_datetime_aliases(
        self, decision_validator: DecisionValidator
    ):
        """DecisionValidator maps slot/time/date aliases to 'meeting_preference'."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="general_inquiry",
            extracted_slots={"slot": "Friday 4 PM"},
            user_facing_response="I noted Friday at 4 PM.",
        )
        validated = decision_validator.validate_and_filter(decision)
        assert validated.extracted_slots.get("meeting_preference") == "Friday 4 PM"

    @pytest.mark.asyncio
    async def test_date_time_extraction_and_persistence_across_turns(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When an inbound caller provides date/time, it is stored in state and persists across turns."""
        mock_llm = MockLLMProvider()
        # Turn 1: Caller says "Can we schedule for today by 12 PM?"
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="schedule_consultation",
                extracted_slots={"meeting_preference": "today by 12 PM"},
                user_facing_response="I can certainly help you schedule that for today by 12 PM.",
            )
        )
        # Turn 2: Caller asks unrelated question "Do you offer voice bots?"
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="solutions_inquiry",
                extracted_slots={},
                user_facing_response="Yes, Vayvora specializes in real-time voice AI agents.",
            )
        )

        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-persist-slot",
            domain=DomainType.VAYVORA,
        )

        # Turn 1
        res1 = await engine.process_user_turn(state, "Can we schedule for today by 12 PM?")
        assert state.get_slot("meeting_preference") == "today by 12 PM"

        # Turn 2: Unrelated topic
        res2 = await engine.process_user_turn(state, "Do you offer voice bots?")
        # Must STILL be present in state
        assert state.get_slot("meeting_preference") == "today by 12 PM"

    @pytest.mark.asyncio
    async def test_calendar_guard_populates_slot_from_state_and_executes(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When meeting_preference is in state, calendar execution does NOT ask again."""
        mock_llm = MockLLMProvider()
        # LLM proposes action without duplicating slot in arguments
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="schedule_consultation",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="create_calendar_event",
                    arguments={},  # Slot missing in arguments!
                ),
                user_facing_response="Scheduling your consultation.",
            )
        )

        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-guard-reuse",
            domain=DomainType.VAYVORA,
        )
        # Pre-existing preference in state
        state.update_slot("meeting_preference", "today by 12 PM")

        turn_result = await engine.process_user_turn(state, "Please confirm the booking.")

        # Guard must NOT ask "Which date or time slot would work best for you?"
        assert "Which date or time slot would work best for you?" not in turn_result.response_text
        # Must execute tool and confirm
        assert state.last_action == "create_calendar_event"
        assert state.last_tool_result.success is True
        assert state.last_tool_result.data.get("slot") == "today by 12 PM"
        assert "today by 12 PM" in turn_result.response_text

    @pytest.mark.asyncio
    async def test_calendar_guard_asks_only_when_no_slot_anywhere(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When NEITHER tool args nor state contain a slot, guard asks caller for slot."""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="schedule_consultation",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="create_calendar_event",
                    arguments={},
                ),
                user_facing_response="I will schedule your consultation.",
            )
        )

        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-guard-ask",
            domain=DomainType.EDUSAAS,
        )

        turn_result = await engine.process_user_turn(state, "I want to schedule a demo.")

        # Guard must intercept and ask for date/time
        assert "Which date or time slot would work best for you?" in turn_result.response_text
        assert state.pending_question == "Which date or time slot would work best for you?"
        # Tool must NOT have been executed yet
        assert state.last_action is None

    @pytest.mark.asyncio
    async def test_caller_answering_pending_slot_question_completes_booking(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """When caller answers pending slot question, slot is captured, tool is executed."""
        mock_llm = MockLLMProvider()
        # MockLLM generates default response without proposed action
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.EDUSAAS,
                detected_intent="schedule_consultation",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                user_facing_response="Great, booking that now.",
            )
        )

        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-answer-slot",
            domain=DomainType.EDUSAAS,
        )
        state.set_pending_question("Which date or time slot would work best for you?")

        turn_result = await engine.process_user_turn(state, "today by 12 PM")

        assert state.get_slot("meeting_preference") == "today by 12 PM"
        assert state.last_action == "create_calendar_event"
        assert state.last_tool_result.success is True
        assert state.last_tool_result.data.get("slot") == "today by 12 PM"
        assert state.pending_question is None

    @pytest.mark.asyncio
    async def test_slot_revision_updates_state_and_subsequent_action(
        self, state_manager: ConversationStateManager, tool_provider: MockToolProvider
    ):
        """If caller says 'actually, make it 2 PM instead', state updates to 2 PM."""
        mock_llm = MockLLMProvider()
        mock_llm.add_decision(
            ConversationalDecision(
                detected_domain=DomainType.VAYVORA,
                detected_intent="schedule_consultation",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                action_proposed=True,
                proposed_action=ProposedAction(
                    tool_name="create_calendar_event",
                    arguments={},
                ),
                user_facing_response="Rescheduling to your preferred time.",
            )
        )

        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)
        state = state_manager.create_inbound_state(
            call_id="call-revision",
            domain=DomainType.VAYVORA,
        )
        # Prior slot was 10 AM
        state.update_slot("meeting_preference", "Tomorrow at 10 AM")

        turn_result = await engine.process_user_turn(
            state, "Actually, make it 2 PM instead."
        )

        assert state.get_slot("meeting_preference") == "2 PM"
        assert state.last_action == "create_calendar_event"
        assert state.last_tool_result.data.get("slot") == "2 PM"
        assert "2 PM" in turn_result.response_text
