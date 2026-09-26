"""Comprehensive unit tests for the MCP Tool Layer & Verified Action Protocol (Phase 5).

Covers:
1. Valid action proposal and schema validation
2. Unsupported action name rejection
3. Malformed/missing arguments (email format, missing recipients, unconfirmed calendar slots)
4. Calendar safety invariants (confirmed slot requirement)
5. Tool execution outcomes (email, calendar availability, calendar booking, lead update, HR followup, message)
6. Central ActionVerifier protocol (verified vs unverified vs rejected)
7. Failure modes: MCP unavailable (503), timeout, auth failure (401), malformed response
8. Idempotency caching preventing duplicate mutations
9. Untrusted tool output & prevention of fabricated success claims
10. Preservation of active conversation after tool failures
11. Safe parameter extraction & conversational clarification when details are missing
"""

import pytest
import asyncio
from typing import Dict, Any

from src.core.types import (
    CallMetadata,
    CallDirection,
    DomainType,
    ToolCallRequest,
    ToolExecutionResult,
)
from src.state.models import ConversationState, CallerProfile
from src.state.manager import ConversationStateManager
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.errors import (
    InvalidToolArgumentsError,
    UnsupportedActionError,
)
from src.tools.schemas import ValidatedToolRequest, ToolResult, VerificationStatus
from src.tools.validation import ActionValidator, SUPPORTED_ACTIONS
from src.tools.verifier import ActionVerifier
from src.tools.mcp_client import MockToolProvider, HttpMCPToolProvider
from src.tools.registry import list_registered_tools, REGISTERED_TOOLS
from src.core.llm import MockLLMProvider


# ============================================================================
# 1. Action Validation & Schema Tests
# ============================================================================

class TestActionValidation:
    """Tests ActionValidator rules and schemas."""

    def test_registered_tools_catalog(self):
        """Verify list of registered tools matches supported actions."""
        tools = list_registered_tools()
        names = {t["name"] for t in tools}
        assert names == SUPPORTED_ACTIONS
        assert "send_email" in names
        assert "create_calendar_event" in names
        assert "find_available_slots" in names
        assert "update_lead" in names
        assert "create_hr_followup" in names
        assert "send_message" in names
        assert "update_business_status" in names

    def test_valid_action_proposal(self):
        """Test valid action proposals pass validation without error."""
        req = ValidatedToolRequest(
            action_name="send_email",
            arguments={"recipient": "user@example.com", "subject": "Info", "body": "Details"},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
        )
        ActionValidator.validate_action(req)

        req_cal = ValidatedToolRequest(
            action_name="create_calendar_event",
            arguments={"slot": "Tomorrow 10:00 AM", "confirmed": True},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
        )
        ActionValidator.validate_action(req_cal)

    def test_unsupported_action_raises_error(self):
        """Test unknown or unsupported action name raises UnsupportedActionError."""
        req = ValidatedToolRequest(
            action_name="execute_system_command",
            arguments={"command": "rm -rf /"},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
        )
        with pytest.raises(UnsupportedActionError) as exc_info:
            ActionValidator.validate_action(req)
        assert "is not supported" in str(exc_info.value)

    def test_email_missing_recipient_raises_error(self):
        """Test send_email without recipient or caller_email raises InvalidToolArgumentsError."""
        req = ValidatedToolRequest(
            action_name="send_email",
            arguments={"subject": "Information"},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
            caller_email=None,
        )
        with pytest.raises(InvalidToolArgumentsError) as exc_info:
            ActionValidator.validate_action(req)
        assert "requires a valid recipient email" in str(exc_info.value)

    def test_email_malformed_recipient_raises_error(self):
        """Test send_email with invalid email format (no @) raises InvalidToolArgumentsError."""
        req = ValidatedToolRequest(
            action_name="send_email",
            arguments={"recipient": "invalid-email-address"},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
        )
        with pytest.raises(InvalidToolArgumentsError) as exc_info:
            ActionValidator.validate_action(req)
        assert "requires a valid recipient email" in str(exc_info.value)

    def test_calendar_creation_without_slot_raises_error(self):
        """Calendar safety: cannot book without specifying a time slot."""
        req = ValidatedToolRequest(
            action_name="create_calendar_event",
            arguments={"confirmed": True},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
        )
        with pytest.raises(InvalidToolArgumentsError) as exc_info:
            ActionValidator.validate_action(req)
        assert "without a specified time slot" in str(exc_info.value)

    def test_calendar_creation_unconfirmed_slot_raises_error(self):
        """Calendar safety: cannot book when confirmed is explicitly False."""
        req = ValidatedToolRequest(
            action_name="create_calendar_event",
            arguments={"slot": "Tomorrow 10:00 AM", "confirmed": False},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
        )
        with pytest.raises(InvalidToolArgumentsError) as exc_info:
            ActionValidator.validate_action(req)
        assert "before caller confirms the slot" in str(exc_info.value)

    def test_send_message_missing_recipient_raises_error(self):
        """send_message requires a phone number or recipient."""
        req = ValidatedToolRequest(
            action_name="send_message",
            arguments={"text": "Hello"},
            session_id="call-001",
            domain=DomainType.VAYVORA,
        )
        with pytest.raises(InvalidToolArgumentsError) as exc_info:
            ActionValidator.validate_action(req)
        assert "requires a recipient phone number" in str(exc_info.value)

    def test_update_lead_empty_arguments_raises_error(self):
        """update_lead requires at least one field."""
        req = ValidatedToolRequest(
            action_name="update_lead",
            arguments={},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
        )
        with pytest.raises(InvalidToolArgumentsError) as exc_info:
            ActionValidator.validate_action(req)
        assert "requires at least one field" in str(exc_info.value)

    def test_create_hr_followup_missing_name_raises_error(self):
        """create_hr_followup requires candidate identification."""
        req = ValidatedToolRequest(
            action_name="create_hr_followup",
            arguments={"position": "Software Engineer"},
            session_id="call-001",
            domain=DomainType.VAYVORA,
        )
        with pytest.raises(InvalidToolArgumentsError) as exc_info:
            ActionValidator.validate_action(req)
        assert "requires candidate name" in str(exc_info.value)

    def test_update_business_status_missing_status_raises_error(self):
        """update_business_status requires a valid status string."""
        req = ValidatedToolRequest(
            action_name="update_business_status",
            arguments={"something": "else"},
            session_id="call-001",
            domain=DomainType.EDUSAAS,
        )
        with pytest.raises(InvalidToolArgumentsError) as exc_info:
            ActionValidator.validate_action(req)
        assert "requires a valid status string" in str(exc_info.value)


# ============================================================================
# 2. Central Action Verification Protocol Tests
# ============================================================================

class TestActionVerifier:
    """Tests the ActionVerifier contract and invariants."""

    def test_verification_success_for_email(self):
        """Email with valid external message_id is marked VERIFIED."""
        raw = ToolResult(
            action_name="send_email",
            succeeded=True,
            data={"status": "sent", "message_id": "msg_gmail_999"},
        )
        verified = ActionVerifier.verify(raw)
        assert verified.is_verified_success is True
        assert verified.verification_status == VerificationStatus.VERIFIED
        assert verified.external_reference == "msg_gmail_999"

    def test_verification_failure_for_email_missing_message_id(self):
        """Email where server returned success but omitted message_id is marked UNVERIFIED."""
        raw = ToolResult(
            action_name="send_email",
            succeeded=True,
            data={"status": "sent"},  # Missing message_id!
        )
        verified = ActionVerifier.verify(raw)
        assert verified.is_verified_success is False
        assert verified.succeeded is False
        assert verified.failed is True
        assert verified.verification_status == VerificationStatus.UNVERIFIED
        assert "missing external message identifier" in verified.error

    def test_verification_success_for_calendar_event(self):
        """Calendar booking with valid event_id is marked VERIFIED."""
        raw = ToolResult(
            action_name="create_calendar_event",
            succeeded=True,
            data={"event_id": "evt_gcal_12345", "slot": "Tomorrow 10 AM"},
        )
        verified = ActionVerifier.verify(raw)
        assert verified.is_verified_success is True
        assert verified.verification_status == VerificationStatus.VERIFIED
        assert verified.external_reference == "evt_gcal_12345"

    def test_verification_failure_for_calendar_event_missing_event_id(self):
        """Calendar booking without event_id is rejected by verifier."""
        raw = ToolResult(
            action_name="create_calendar_event",
            succeeded=True,
            data={"status": "confirmed"},  # Missing event_id!
        )
        verified = ActionVerifier.verify(raw)
        assert verified.is_verified_success is False
        assert verified.verification_status == VerificationStatus.UNVERIFIED
        assert "missing external event reference" in verified.error

    def test_verification_success_for_available_slots(self):
        """find_available_slots returning list of slots is VERIFIED."""
        raw = ToolResult(
            action_name="find_available_slots",
            succeeded=True,
            data={"slots": ["Tomorrow 10:00 AM", "Tomorrow 2:00 PM"]},
        )
        verified = ActionVerifier.verify(raw)
        assert verified.is_verified_success is True
        assert verified.verification_status == VerificationStatus.VERIFIED
        assert verified.external_reference == "slots_found_2"

    def test_verification_failure_for_available_slots_invalid_payload(self):
        """find_available_slots without list is UNVERIFIED."""
        raw = ToolResult(
            action_name="find_available_slots",
            succeeded=True,
            data={"slots": "not-a-list"},
        )
        verified = ActionVerifier.verify(raw)
        assert verified.is_verified_success is False
        assert verified.verification_status == VerificationStatus.UNVERIFIED

    def test_verification_for_already_failed_action(self):
        """Actions that failed at transport or server level stay FAILED."""
        raw = ToolResult(
            action_name="send_email",
            succeeded=False,
            failed=True,
            error="Connection timeout",
        )
        verified = ActionVerifier.verify(raw)
        assert verified.is_verified_success is False
        assert verified.verification_status == VerificationStatus.FAILED
        assert verified.failed is True


# ============================================================================
# 3. Mock MCP Client Execution Tests (All Actions & Failure Modes)
# ============================================================================

class TestMockToolProvider:
    """Tests MockToolProvider with all actions, failure modes, and idempotency."""

    @pytest.mark.asyncio
    async def test_successful_email_dispatch(self):
        """send_email dispatches and returns verified result."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="send_email",
            arguments={"recipient": "lead@example.com", "subject": "Curriculum", "body": "Attached"},
            call_id="call-email-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is True
        assert res.verification_code is not None
        assert "msg_mock_" in res.verification_code
        assert res.data.get("status") == "dispatched"

    @pytest.mark.asyncio
    async def test_find_available_slots(self):
        """find_available_slots returns available slots list."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="find_available_slots",
            arguments={"date": "tomorrow"},
            call_id="call-slots-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is True
        assert "slots" in res.data
        assert len(res.data["slots"]) >= 1

    @pytest.mark.asyncio
    async def test_calendar_creation_with_confirmed_slot(self):
        """create_calendar_event with confirmed slot succeeds."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="create_calendar_event",
            arguments={"slot": "Tomorrow 10:00 AM", "confirmed": True},
            call_id="call-cal-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is True
        assert res.verification_code == "evt_cal_987654"
        assert res.data.get("status") == "confirmed"

    @pytest.mark.asyncio
    async def test_calendar_creation_without_confirmed_slot_fails_validation(self):
        """create_calendar_event with confirmed=False fails pre-execution validation."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="create_calendar_event",
            arguments={"slot": "Tomorrow 10:00 AM", "confirmed": False},
            call_id="call-cal-02",
        )
        res = await provider.execute_tool(req)
        assert res.success is False
        assert "before caller confirms the slot" in res.error_message
        assert len(provider.dispatched_requests) == 0

    @pytest.mark.asyncio
    async def test_lead_update_success(self):
        """update_lead updates CRM and returns lead_id."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="update_lead",
            arguments={"lead_status": "interested", "notes": "Wants full syllabus"},
            call_id="call-lead-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is True
        assert res.verification_code == "lead_crm_12345"

    @pytest.mark.asyncio
    async def test_create_hr_followup_success(self):
        """create_hr_followup queues HR ticket and returns ticket_id."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="create_hr_followup",
            arguments={"candidate_name": "Jane Doe", "role": "Senior Engineer"},
            call_id="call-hr-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is True
        assert res.verification_code == "ticket_hr_554433"
        assert res.data.get("status") == "queued"

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """send_message dispatches SMS and returns message_id."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="send_message",
            arguments={"recipient": "+15551234567", "text": "Confirmation SMS"},
            call_id="call-sms-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is True
        assert res.verification_code == "msg_sms_778899"

    @pytest.mark.asyncio
    async def test_update_business_status_success(self):
        """update_business_status updates state and returns verified status."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="update_business_status",
            arguments={"status": "qualified_lead"},
            call_id="call-bus-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is True
        assert res.verification_code == "status_qualified_lead"

    @pytest.mark.asyncio
    async def test_mcp_unavailable_simulation(self):
        """Provider simulating 503 HTTP unreachable fails cleanly."""
        provider = MockToolProvider(force_unavailable=True)
        req = ToolCallRequest(
            tool_name="send_email",
            arguments={"recipient": "user@example.com"},
            call_id="call-err-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is False
        assert "unreachable" in res.error_message
        assert res.verification_code is None

    @pytest.mark.asyncio
    async def test_mcp_timeout_simulation(self):
        """Provider simulating timeout returns failed result with timeout error."""
        provider = MockToolProvider(force_timeout=True)
        req = ToolCallRequest(
            tool_name="find_available_slots",
            arguments={},
            call_id="call-err-02",
        )
        res = await provider.execute_tool(req)
        assert res.success is False
        assert "timed out" in res.error_message

    @pytest.mark.asyncio
    async def test_mcp_auth_failure_simulation(self):
        """Provider simulating 401 auth failure returns failed result."""
        provider = MockToolProvider(force_auth_failure=True)
        req = ToolCallRequest(
            tool_name="update_lead",
            arguments={"status": "follow_up"},
            call_id="call-err-03",
        )
        res = await provider.execute_tool(req)
        assert res.success is False
        assert "authentication failed" in res.error_message

    @pytest.mark.asyncio
    async def test_mcp_malformed_response_simulation(self):
        """Provider simulating malformed response handles error without crash."""
        provider = MockToolProvider(force_malformed=True)
        req = ToolCallRequest(
            tool_name="send_email",
            arguments={"recipient": "user@example.com"},
            call_id="call-err-04",
        )
        res = await provider.execute_tool(req)
        assert res.success is False
        assert "Malformed response" in res.error_message

    @pytest.mark.asyncio
    async def test_verification_failure_simulation(self):
        """When MCP server reports success but omits external ID, result is converted to failure."""
        provider = MockToolProvider(force_verification_failure=True)
        req = ToolCallRequest(
            tool_name="send_email",
            arguments={"recipient": "user@example.com"},
            call_id="call-ver-01",
        )
        res = await provider.execute_tool(req)
        assert res.success is False
        assert "missing external message identifier" in res.error_message

    @pytest.mark.asyncio
    async def test_idempotency_caching_prevents_duplicate_mutations(self):
        """Duplicate request with same arguments uses idempotency cache and avoids repeated dispatch."""
        provider = MockToolProvider()
        req = ToolCallRequest(
            tool_name="send_email",
            arguments={"recipient": "lead@example.com", "subject": "Hello"},
            call_id="call-idem-01",
        )

        res1 = await provider.execute_tool(req)
        assert res1.success is True
        assert len(provider.dispatched_requests) == 1

        # Second identical execution
        res2 = await provider.execute_tool(req)
        assert res2.success is True
        # Dispatched requests count must STILL be 1 (served from cache)
        assert len(provider.dispatched_requests) == 1
        assert res1.verification_code == res2.verification_code


# ============================================================================
# 4. ConversationEngine & Verified Action Protocol Integration Tests
# ============================================================================

class TestEngineToolIntegration:
    """Tests ConversationEngine tool execution, guards, and non-terminating error handling."""

    @pytest.mark.asyncio
    async def test_successful_action_updates_state_and_synthesizes_confirmation(self):
        """When tool succeeds, engine completes action in state and synthesizes confirmation."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="request_brochure",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "lead@example.com", "subject": "Brochure"},
            ),
            user_facing_response="I will send that over.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="engine-tool-01",
            caller_phone="+1234567890",
            domain=DomainType.EDUSAAS,
        )

        turn_result = await engine.process_user_turn(
            state=state,
            user_message="Please email me the brochure to lead@example.com.",
        )

        assert state.pending_action is None
        assert state.last_tool_result is not None
        assert state.last_tool_result.success is True
        assert state.last_tool_result.verification_code is not None
        assert state.last_action == "send_email"
        assert state.conversation_active is True
        assert len(state.history) == 2

    @pytest.mark.asyncio
    async def test_email_missing_recipient_asks_caller_and_avoids_provider_call(self):
        """Conversational guard: if caller email is unknown and not in args, engine asks for email."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="request_email",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"subject": "Course Catalog"},  # No recipient!
            ),
            user_facing_response="I'll email it.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="engine-email-guard-01",
            caller_phone="+1234567890",
            domain=DomainType.EDUSAAS,
        )

        turn_result = await engine.process_user_turn(
            state=state,
            user_message="Could you email me the catalog?",
        )

        # Provider must NOT have been called
        assert len(tool_provider.dispatched_requests) == 0
        # Engine asked for email
        assert "email address" in turn_result.response_text.lower()
        assert state.pending_question == "Could you please share your email address?"
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_calendar_missing_slot_asks_caller_and_avoids_provider_call(self):
        """Calendar guard: if slot is missing, engine asks caller which time slot works best."""
        decision = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="schedule_consultation",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="create_calendar_event",
                arguments={"confirmed": True},  # Missing slot!
            ),
            user_facing_response="I will book that.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="engine-cal-guard-01",
            caller_phone="+1234567890",
            domain=DomainType.VAYVORA,
        )

        turn_result = await engine.process_user_turn(
            state=state,
            user_message="I'd like to book a meeting.",
        )

        assert len(tool_provider.dispatched_requests) == 0
        assert "time slot" in turn_result.response_text.lower()
        assert state.pending_question == "Which date or time slot would work best for you?"
        assert state.conversation_active is True

    @pytest.mark.asyncio
    async def test_action_failure_does_not_terminate_conversation(self):
        """When tool fails, conversation remains active, error is logged, and caller is not disconnected."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="send_brochure",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "test@example.com"},
            ),
            user_facing_response="Sending email now.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider(force_unavailable=True)
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="engine-fail-01",
            caller_phone="+1234567890",
            domain=DomainType.EDUSAAS,
        )

        turn_result = await engine.process_user_turn(
            state=state,
            user_message="Please send the syllabus to test@example.com.",
        )

        # Call must remain active
        assert state.conversation_active is True
        # Tool failure recorded
        assert state.last_tool_result is not None
        assert state.last_tool_result.success is False
        assert state.last_action != "send_email"
        # Polite failure explanation presented to caller
        assert "system issue" in turn_result.response_text.lower()

    @pytest.mark.asyncio
    async def test_verification_failure_prevents_fabricated_success_speech(self):
        """When tool reports HTTP success but verification fails, engine does NOT claim success."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="book_consultation",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="create_calendar_event",
                arguments={"slot": "Tomorrow 3:00 PM", "confirmed": True},
            ),
            user_facing_response="Booking that right now.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        # Server says OK, but omits event reference
        tool_provider = MockToolProvider(force_verification_failure=True)
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="engine-ver-fail-01",
            caller_phone="+1234567890",
            domain=DomainType.EDUSAAS,
        )

        turn_result = await engine.process_user_turn(
            state=state,
            user_message="Confirm Tomorrow at 3 PM for my consultation.",
        )

        assert state.conversation_active is True
        assert state.last_tool_result is not None
        assert state.last_tool_result.success is False
        # Does not record as completed action
        assert state.last_action != "create_calendar_event"
        # Response should reflect system issue, NOT fabricated confirmation
        assert "system issue" in turn_result.response_text.lower()

    @pytest.mark.asyncio
    async def test_auto_populate_caller_profile_details(self):
        """Engine auto-populates known caller profile attributes into tool call arguments."""
        decision = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="send_brochure",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"subject": "Curriculum PDF"},  # recipient omitted in proposed_action
            ),
            user_facing_response="Sending to your registered email.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="engine-auto-pop-01",
            caller_phone="+1234567890",
            domain=DomainType.EDUSAAS,
            caller_name="Alice Smith",
            caller_email="alice@university.edu",
            campaign_id="camp-01",
            campaign_objective="Syllabus follow up",
        )

        turn_result = await engine.process_user_turn(
            state=state,
            user_message="Go ahead and email that to me.",
        )

        # Provider received auto-populated email
        assert len(tool_provider.dispatched_requests) == 1
        dispatched_req = tool_provider.dispatched_requests[0]
        assert dispatched_req.arguments.get("recipient") == "alice@university.edu"
        assert dispatched_req.arguments.get("caller_name") == "Alice Smith"
        assert state.last_tool_result.success is True

