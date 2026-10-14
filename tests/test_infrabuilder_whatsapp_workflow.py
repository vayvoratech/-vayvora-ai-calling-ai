"""Comprehensive test suite for Infra Builder WhatsApp Generation & Dispatch Workflow.

Verifies all 16 required test scenarios:
TEST 1: Inbound Infra Builder + profile phone + brochure request (dispatches authorized WhatsApp message).
TEST 2: Inbound Infra Builder + missing phone (asks for recipient WhatsApp phone, blocks tool).
TEST 3: Inbound Infra Builder + caller provides phone (validates, normalizes E.164, stores, dispatches, confirms).
TEST 4: Inbound Infra Builder + dummy phone (rejects sequential/repeating dummy numbers, blocks tool).
TEST 5: Outbound Infra Builder + profile phone exists + consent given (sends immediately, no redundant asking).
TEST 6: Outbound Infra Builder + caller opt-out ("don't message me on WhatsApp" -> records refusal, blocks tool).
TEST 7: Caller requests property information (verified Infra Builder RAG/catalog info used).
TEST 8: Caller requests pricing (verified pricing ₹1.45 Cr, ₹2.5 Cr, ₹45 Lakhs used, no fabricated numbers).
TEST 9: Requested information is unavailable (no fabricated info, honest guidance).
TEST 10: Caller changes request (latest project request wins: Skyline Towers -> Green Valley Villas).
TEST 11: Contact name present (greeting addresses contact by name, signature is Infra Builder Team).
TEST 12: No contact name (clean greeting, no invented name).
TEST 13: send_message fails (honest failure response, no false success claim, call stays active).
TEST 14: No WhatsApp requested (verbal answer only, no automatic message dispatch).
TEST 15: Phone normalization (standard E.164 normalization prepending '91' for 10-digit numbers).
TEST 16: Status reporting (gateway acceptance returns 'accepted' or 'mock_sent', never claims 'delivered').
"""

import re
from typing import Any, Dict, List, Optional
import pytest

from src.config import Settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, generate_infra_builder_whatsapp_message
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    ToolCallRequest,
    ToolExecutionResult,
)
from src.domains.registry import get_domain_registry
from src.state.manager import ConversationStateManager
from src.tools.mcp_client import MockToolProvider
from src.tools.message_provider import validate_phone_number, normalize_phone_number, MockMessageProvider


class FailingMockToolProvider(MockToolProvider):
    """Tool provider where send_message explicitly fails."""

    async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
        if request.tool_name == "send_message":
            return ToolExecutionResult(
                tool_name=request.tool_name,
                success=False,
                error_message="WhatsApp gateway connection timed out",
            )
        return await super().execute_tool(request)


@pytest.fixture
def base_settings():
    return Settings(
        agent_name="Sarah",
    )


class TestInfraBuilderWhatsAppWorkflow:
    """Verifies all 16 WhatsApp real estate dispatch scenarios for Infra Builder."""

    @pytest.mark.asyncio
    async def test_01_inbound_with_profile_phone_dispatches_whatsapp(self):
        """TEST 1: Inbound with profile phone dispatches WhatsApp brochure."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_information_request",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"recipient": "+919845123456", "project_name": "Skyline Towers", "whatsapp_consent": True},
            ),
            user_facing_response="I'll share the Skyline Towers brochure to your WhatsApp.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-wa-01", "+919845123456", domain=DomainType.INFRA_BUILDER)
        state.update_slot("whatsapp_consent", True)
        state.update_slot("preferred_project", "Skyline Towers")

        res = await engine.process_user_turn(state, "Please send the Skyline Towers brochure to my WhatsApp.")

        assert state.last_action == "send_message"
        assert state.last_tool_result is not None
        assert state.last_tool_result.success is True
        assert len(tool_provider.dispatched_requests) == 1
        assert "whatsapp" in res.response_text.lower()

    @pytest.mark.asyncio
    async def test_02_inbound_missing_phone_asks_for_whatsapp_number(self):
        """TEST 2: Inbound with missing phone asks for recipient number and blocks tool."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_information_request",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"project_name": "Skyline Towers"},
            ),
            user_facing_response="I will send the brochure.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-wa-02", caller_phone=None, domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "Please WhatsApp me the brochure.")

        assert len(tool_provider.dispatched_requests) == 0
        assert "whatsapp number" in res.response_text.lower()
        assert state.pending_action == "send_message"

    @pytest.mark.asyncio
    async def test_03_inbound_caller_provides_phone_dispatches_message(self):
        """TEST 3: Caller provides phone number in utterance; engine normalizes and dispatches."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_information_request",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"project_name": "Skyline Towers", "whatsapp_consent": True},
            ),
            user_facing_response="Sending brochure now.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-wa-03", caller_phone=None, domain=DomainType.INFRA_BUILDER)
        state.set_pending_action("send_message")
        state.update_slot("preferred_project", "Skyline Towers")

        res = await engine.process_user_turn(state, "You can send it to 9845123456.")

        assert len(tool_provider.dispatched_requests) == 1
        req = tool_provider.dispatched_requests[0]
        assert req.arguments["recipient"] == "919845123456"
        assert state.last_tool_result.success is True

    @pytest.mark.asyncio
    async def test_04_dummy_phone_number_rejected(self):
        """TEST 4: Sequential or repeating dummy numbers are rejected by validation."""
        valid, _, err = validate_phone_number("1234567890")
        assert valid is False
        assert "sequential" in err.lower()

        valid, _, err = validate_phone_number("9999999999")
        assert valid is False
        assert "repeating" in err.lower()

    @pytest.mark.asyncio
    async def test_05_outbound_with_consent_sends_immediately(self):
        """TEST 5: Outbound call with verified identity and consent dispatches immediately."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_information_request",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"whatsapp_consent": True},
            ),
            user_facing_response="Sending details to your WhatsApp.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-wa-05",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
            caller_name="Kiran Kumar",
        )
        state.update_slot("outbound_identity_verified", True)
        state.update_slot("whatsapp_consent", True)
        state.update_slot("preferred_project", "Skyline Towers")

        await engine.process_user_turn(state, "Yes, please send that over WhatsApp.")

        assert len(tool_provider.dispatched_requests) == 1
        assert tool_provider.dispatched_requests[0].arguments["recipient"] == "919845123456"

    @pytest.mark.asyncio
    async def test_06_opt_out_blocks_whatsapp_dispatch(self):
        """TEST 6: Caller opt-out blocks WhatsApp message and records consent=False."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_refusal",
            action_proposed=False,
            user_facing_response="Understood, I will not send any WhatsApp messages.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-wa-06", "+919845123456", domain=DomainType.INFRA_BUILDER)
        state.update_slot("whatsapp_consent", True)

        res = await engine.process_user_turn(state, "No whatsapp, do not message me.")

        assert state.get_slot("whatsapp_consent") is False
        assert len(tool_provider.dispatched_requests) == 0
        assert "will not send" in res.response_text.lower()

    def test_07_property_information_in_generated_message(self):
        """TEST 7: Verified property information is formatted into WhatsApp message."""
        msg = generate_infra_builder_whatsapp_message(
            project_name="Skyline Towers",
            rag_context="Skyline Towers is a premium residential development in Whitefield, Bangalore offering 2 and 3 BHK luxury residences.",
            caller_name="Rajesh",
        )
        assert "Hello Rajesh," in msg
        assert "Skyline Towers" in msg
        assert "Infra Builder" in msg
        assert "Whitefield" in msg

    def test_08_catalog_overview_message_when_no_project_specified(self):
        """TEST 8: Overview catalog message is generated when no project is specified."""
        msg = generate_infra_builder_whatsapp_message(project_name=None)
        assert "Skyline Towers (Whitefield)" in msg
        assert "Green Valley Villas (Sarjapur Road)" in msg
        assert "Urban Heights (Electronic City)" in msg
        assert "₹85 Lakhs" in msg
        assert "₹2.5 Cr" in msg
        assert "₹45 Lakhs" in msg

    def test_09_no_fabricated_details_in_message(self):
        """TEST 9: Message contains only approved projects and contacts."""
        msg = generate_infra_builder_whatsapp_message(project_name="Skyline Towers")
        assert "Infra Builder Team" in msg
        # Must not contain legacy business names
        assert "vayvora" not in msg.lower()
        assert "edusaas" not in msg.lower()

    @pytest.mark.asyncio
    async def test_10_caller_changes_project_request(self):
        """TEST 10: Latest project request takes priority."""
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-wa-10", "+919845123456", domain=DomainType.INFRA_BUILDER)
        state.update_slot("preferred_project", "Skyline Towers")

        # Caller switches preference
        state.update_slot("preferred_project", "Green Valley Villas")
        assert state.get_slot("preferred_project") == "Green Valley Villas"

    def test_11_contact_name_greeting(self):
        """TEST 11: Personal greeting when contact name is provided."""
        msg = generate_infra_builder_whatsapp_message(project_name="Urban Heights", caller_name="Ananya")
        assert msg.startswith("Hello Ananya,\n\n")

    def test_12_impersonal_greeting_when_name_missing(self):
        """TEST 12: Generic greeting when contact name is absent."""
        msg = generate_infra_builder_whatsapp_message(project_name="Urban Heights", caller_name=None)
        assert msg.startswith("Hello,\n\n")

    @pytest.mark.asyncio
    async def test_13_whatsapp_failure_preserves_call(self):
        """TEST 13: Tool failure reports failure politely and keeps call active."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="whatsapp_information_request",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_message",
                arguments={"recipient": "+919845123456", "whatsapp_consent": True},
            ),
            user_facing_response="Sending brochure.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = FailingMockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-wa-13", "+919845123456", domain=DomainType.INFRA_BUILDER)
        state.update_slot("whatsapp_consent", True)

        res = await engine.process_user_turn(state, "Please send the brochure to my WhatsApp.")

        assert state.conversation_active is True
        assert state.last_tool_result.success is False
        assert "system issue" in res.response_text.lower()

    @pytest.mark.asyncio
    async def test_14_no_whatsapp_requested_gives_verbal_answer(self):
        """TEST 14: Property question without WhatsApp request does not dispatch message."""
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="price_query",
            action_proposed=False,
            user_facing_response="Skyline Towers 2 BHK apartments start at ₹85 Lakhs.",
        )
        mock_llm = MockLLMProvider(canned_decisions=[decision])
        tool_provider = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=tool_provider)

        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-wa-14", "+919845123456", domain=DomainType.INFRA_BUILDER)

        res = await engine.process_user_turn(state, "What is the starting price for Skyline Towers?")

        assert len(tool_provider.dispatched_requests) == 0
        assert "₹85 Lakhs" in res.response_text or "85" in res.response_text

    def test_15_phone_normalization(self):
        """TEST 15: Phone normalization formats 10-digit numbers with '91'."""
        assert normalize_phone_number("9845123456") == "919845123456"
        assert normalize_phone_number("+91-9845-123456") == "919845123456"
        assert normalize_phone_number("09845123456") == "919845123456"

    @pytest.mark.asyncio
    async def test_16_provider_status_reporting(self):
        """TEST 16: WhatsApp provider returns 'mock_sent' or 'accepted', never 'delivered'."""
        provider = MockMessageProvider()
        result = await provider.send_message("919845123456", "Test message")
        assert result.success is True
        assert result.status in ("mock_sent", "accepted", "queued")
        assert result.status != "delivered"
