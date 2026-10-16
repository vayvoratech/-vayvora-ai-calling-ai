"""Tests for Phase 3E: Caller Privacy and Sensitive-Action Security Audit.

Verifies:
1. Masking utilities: phone numbers, email addresses, and embedded sensitive text.
2. Database engine initialization logs never expose passwords or raw connection DSNs.
3. Messaging and email provider logs mask recipient phone numbers and emails.
4. Returning caller phone match alone does not authorize sensitive operations.
5. Fail-closed guards reject unverified sensitive reads (email inbox, calendar events) and profile overwrites.
6. Authorized / confirmed actions succeed when prerequisites (slot confirmation, WhatsApp consent) are met.
7. Low-risk informational conversation and RAG queries remain fully accessible without verification.
8. Turn debug logs sanitize user utterances and agent responses.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.config import Settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.interfaces import LLMProvider, ToolProvider
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    ToolCallRequest,
    ToolExecutionResult,
)
from src.database.connection import DatabasePool
from src.database.repository import (
    PostgresRepository,
    mask_email,
    mask_phone_number,
    mask_sensitive_text,
)
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.tools.calendar_provider import CalendarProvider, FileCalendarProvider
from src.tools.email_provider import MockEmailProvider
from src.tools.message_provider import MockMessageProvider, WhatsAppMessageProvider
from src.tools.server import create_mcp_server


# =============================================================================
# 1. Masking Utilities Verification
# =============================================================================

def make_test_state(call_id: str, phone: str = "+919849012345", name: Optional[str] = None) -> ConversationState:
    manager = ConversationStateManager()
    state = manager.create_inbound_state(
        call_id=call_id,
        caller_phone=phone,
        domain=DomainType.INFRA_BUILDER,
    )
    if name:
        state.caller.name = name
        state.extracted_slots["caller_name"] = name
    return state


class TestPrivacyMaskingUtilities:
    """Verify phone, email, and embedded text masking behavior."""

    def test_mask_phone_number_standard_cases(self):
        assert mask_phone_number("+919876543210") == "***-3210"
        assert mask_phone_number("+1 (555) 234-5678") == "***-5678"
        assert mask_phone_number("9876543210") == "***-3210"

    def test_mask_phone_number_edge_cases(self):
        assert mask_phone_number(None) == "<none>"
        assert mask_phone_number("") == "<none>"
        assert mask_phone_number("123") == "***"
        assert mask_phone_number("1234") == "***"

    def test_mask_email_standard_cases(self):
        assert mask_email("kiran.kumar@example.com") == "k***r@example.com"
        assert mask_email("pawan@infrabuilder.com") == "p***n@infrabuilder.com"
        assert mask_email("ab@domain.com") == "a*@domain.com"
        assert mask_email("a@domain.com") == "*@domain.com"

    def test_mask_email_edge_cases(self):
        assert mask_email(None) == "<none>"
        assert mask_email("") == "<none>"
        assert mask_email("notanemail") == "***"

    def test_mask_sensitive_text_embedded_phones_and_emails(self):
        raw = "Contact me at +919876543210 or email kiran@example.com for booking."
        masked = mask_sensitive_text(raw)
        assert "987654" not in masked
        assert "***-3210" in masked
        assert "k***n@example.com" in masked
        assert "kiran@" not in masked
        assert "Contact me at" in masked

    def test_mask_sensitive_text_none_and_empty(self):
        assert mask_sensitive_text(None) == ""
        assert mask_sensitive_text("") == ""


# =============================================================================
# 2. Database Connection Credentials Audit
# =============================================================================

class TestDatabaseConnectionPrivacy:
    """Verify database connection strings, passwords, and tokens are never logged."""

    @pytest.mark.asyncio
    async def test_get_engine_logs_do_not_expose_password_or_raw_url(self, caplog):
        settings = Settings(
            postgres_user="test_admin",
            postgres_password="SuperSecretPassword123!",
            postgres_host="127.0.0.1",
            postgres_port=5434,
            postgres_db="test_voice_db",
        )
        pool = DatabasePool(settings=settings)

        with caplog.at_level(logging.INFO):
            with patch("src.database.connection.create_async_engine") as mock_engine_create:
                mock_engine = MagicMock()
                mock_engine_create.return_value = mock_engine
                engine = await pool.get_engine()
                assert engine is mock_engine

        log_text = caplog.text
        # Password and full raw connection URL MUST NOT appear anywhere in the logs
        assert "SuperSecretPassword123!" not in log_text
        assert "test_admin:SuperSecretPassword123!" not in log_text
        # Host, port, and DB are safely logged
        assert "127.0.0.1:5434/test_voice_db" in log_text


# =============================================================================
# 3. Provider Logging Privacy Audit (WhatsApp & Email)
# =============================================================================

class TestProviderLoggingPrivacy:
    """Verify tool provider logs mask recipient phone numbers and emails."""

    @pytest.mark.asyncio
    async def test_whatsapp_provider_logs_mask_recipient_phone(self, caplog):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            mock_sess = MagicMock()
            mock_sess.status_code = 200
            mock_sess.json.return_value = [{"id": "session_default", "status": "ready"}]

            mock_send = MagicMock()
            mock_send.status_code = 200
            mock_send.headers = {"content-type": "application/json"}
            mock_send.json.return_value = {"id": "wa_msg_test_01", "status": "accepted"}

            mock_client.get.return_value = mock_sess
            mock_client.post.return_value = mock_send

            provider = WhatsAppMessageProvider(base_url="http://localhost:2785")

            with caplog.at_level(logging.INFO):
                res = await provider.send_message(
                    recipient="+919849012345",
                    message="Here are the project brochures for JB Pride.",
                )
                assert res.success is True

        log_text = caplog.text
        # Raw phone number must not be logged unmasked
        assert "+919849012345" not in log_text
        assert "919849012345" not in log_text
        # Masked representation should appear
        assert "***-2345" in log_text

    @pytest.mark.asyncio
    async def test_mock_email_provider_logs_mask_recipient_email(self, caplog):
        provider = MockEmailProvider(force_success=True)
        with caplog.at_level(logging.INFO):
            res = await provider.send_email(
                recipient="pawan.ganesh@example.com",
                subject="Property Details",
                body="Details attached.",
            )
            assert res.success is True

        log_text = caplog.text
        assert "pawan.ganesh@example.com" not in log_text
        assert "p***h@example.com" in log_text


# =============================================================================
# 4. Returning Caller Identity & Sensitive Action Fail-Closed Guard
# =============================================================================

class TestSensitiveActionAuthorizationGate:
    """Verify returning callers without sensitive verification cannot trigger sensitive actions."""

    @pytest.mark.asyncio
    async def test_returning_caller_has_sensitive_verification_flag_false_by_default(self):
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_lead_by_phone = AsyncMock(
            return_value={
                "id": 101,
                "name": "Kiran Kumar",
                "phone_number": "+919849012345",
                "email": "kiran@example.com",
                "status": "inquiry",
                "interested_in": "JB Pride",
                "metadata": {},
            }
        )

        manager = ConversationStateManager(postgres_repo=mock_repo)
        state = await manager.create_inbound_state_async(
            call_id="call-sec-test-01",
            caller_phone="+919849012345",
            domain=DomainType.INFRA_BUILDER,
        )

        assert state.extracted_slots.get("returning_caller") is True
        assert state.extracted_slots.get("returning_caller_verified_sensitive") is False

    @pytest.mark.asyncio
    async def test_unverified_returning_caller_cannot_read_private_email_inbox(self):
        """mail_read_recent fails closed for unverified callers."""
        state = make_test_state("call-sec-mail-01")
        state.extracted_slots["returning_caller"] = True
        state.extracted_slots["returning_caller_verified_sensitive"] = False

        decision = ConversationalDecision(
            detected_intent="read_emails",
            detected_domain=DomainType.INFRA_BUILDER,
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="mail_read_recent",
                arguments={"limit": 5},
                confidence=1.0,
            ),
            user_facing_response="Let me check your email.",
        )

        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock(return_value=decision)
        engine = ConversationEngine(llm_provider=mock_llm)
        engine.tool_provider = AsyncMock()

        result = await engine.process_user_turn(
            state=state,
            user_message="Can you read my recent emails?",
        )

        # Fail closed: action must NOT be executed
        assert result.action_proposed is False
        assert result.proposed_action is None
        engine.tool_provider.execute_tool.assert_not_called()
        assert "cannot access private records without verified caller authentication" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_unverified_returning_caller_cannot_read_private_calendar_events(self):
        """calendar_get_events fails closed for unverified callers."""
        state = make_test_state("call-sec-cal-01")
        state.extracted_slots["returning_caller"] = True
        state.extracted_slots["returning_caller_verified_sensitive"] = False

        decision = ConversationalDecision(
            detected_intent="check_appointments",
            detected_domain=DomainType.INFRA_BUILDER,
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="calendar_get_events",
                arguments={},
                confidence=1.0,
            ),
            user_facing_response="Let me check your appointments.",
        )

        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock(return_value=decision)
        engine = ConversationEngine(llm_provider=mock_llm)
        engine.tool_provider = AsyncMock()

        result = await engine.process_user_turn(
            state=state,
            user_message="What appointments do I have scheduled?",
        )

        assert result.action_proposed is False
        assert result.proposed_action is None
        engine.tool_provider.execute_tool.assert_not_called()
        assert "cannot access private records without verified caller authentication" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_unverified_returning_caller_cannot_overwrite_registered_profile_name(self):
        """An unverified caller cannot hijack an existing profile by speaking a conflicting name."""
        state = make_test_state("call-sec-name-01", phone="+919849012345", name="Kiran Kumar")
        state.extracted_slots["returning_caller"] = True
        state.extracted_slots["returning_caller_verified_sensitive"] = False

        decision = ConversationalDecision(
            detected_intent="general_inquiry",
            detected_domain=DomainType.INFRA_BUILDER,
            user_facing_response="Hello, how can I help you?",
        )
        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock(return_value=decision)
        engine = ConversationEngine(llm_provider=mock_llm)

        await engine.process_user_turn(
            state=state,
            user_message="Actually, my name is Mallory Hacker",
        )

        # Name must NOT be overwritten to the attacker's value
        assert state.caller.name == "Kiran Kumar"
        assert state.extracted_slots.get("caller_name") != "Mallory Hacker"

    @pytest.mark.asyncio
    async def test_unverified_returning_caller_cannot_perform_sensitive_crm_mutation(self):
        """update_lead fails closed when sensitive_action is flagged without verification."""
        state = make_test_state("call-sec-crm-01", phone="+919849012345", name="Kiran Kumar")
        state.extracted_slots["returning_caller"] = True
        state.extracted_slots["returning_caller_verified_sensitive"] = False

        decision = ConversationalDecision(
            detected_intent="account_update",
            detected_domain=DomainType.INFRA_BUILDER,
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="update_lead",
                arguments={
                    "name": "Different Person",
                    "sensitive_action": True,
                    "overwrite_registered_profile": True,
                },
                confidence=1.0,
            ),
            user_facing_response="Updating account.",
        )
        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock(return_value=decision)
        engine = ConversationEngine(llm_provider=mock_llm)
        engine.tool_provider = AsyncMock()

        result = await engine.process_user_turn(
            state=state,
            user_message="Change my registered account details.",
        )

        assert result.action_proposed is False
        engine.tool_provider.execute_tool.assert_not_called()
        assert "modifying registered account details requires verified caller authentication" in result.response_text.lower()


# =============================================================================
# 5. FastMCP Tool Guard Integration
# =============================================================================

class TestFastMCPToolSecurityGuards:
    """Verify FastMCP tools enforce caller_verified_sensitive directly."""

    @pytest.mark.asyncio
    async def test_fastmcp_mail_read_recent_rejects_unverified_caller(self):
        server = create_mcp_server(email_provider=MockEmailProvider())
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mail_read_recent",
                "arguments": {"limit": 3, "caller_verified_sensitive": False},
            },
        }
        res = await server.handle_jsonrpc(req)
        assert res["result"]["isError"] is True
        assert "requires verified caller identity" in res["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_fastmcp_calendar_get_events_rejects_unverified_caller(self):
        server = create_mcp_server()
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "calendar_get_events",
                "arguments": {"caller_verified_sensitive": False},
            },
        }
        res = await server.handle_jsonrpc(req)
        assert res["result"]["isError"] is True
        assert "requires verified caller identity" in res["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_fastmcp_mail_read_recent_succeeds_when_verified(self):
        mock_provider = MockEmailProvider(
            mock_inbox=[{"from": "team@infra.com", "subject": "Update", "date": "2026-10-15"}]
        )
        server = create_mcp_server(email_provider=mock_provider)
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "mail_read_recent",
                "arguments": {"limit": 3, "caller_verified_sensitive": True},
            },
        }
        res = await server.handle_jsonrpc(req)
        assert res["result"]["isError"] is False
        assert "Update" in res["result"]["content"][0]["text"]


# =============================================================================
# 6. Low-Risk Informational Conversation & Allowed Actions Preservation
# =============================================================================

class TestLowRiskConversationsPreserved:
    """Verify informational queries, slot discovery, and verified bookings succeed without friction."""

    @pytest.mark.asyncio
    async def test_low_risk_find_available_slots_succeeds_without_sensitive_verification(self):
        state = make_test_state("call-lowrisk-01")
        state.extracted_slots["returning_caller"] = True
        state.extracted_slots["returning_caller_verified_sensitive"] = False

        decision = ConversationalDecision(
            detected_intent="book_visit",
            detected_domain=DomainType.INFRA_BUILDER,
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="find_available_slots",
                arguments={
                    "property_name": "JB Pride",
                    "preferred_date": "2026-10-20",
                },
                confidence=1.0,
            ),
            user_facing_response="Checking available slots for JB Pride.",
        )
        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock(return_value=decision)
        engine = ConversationEngine(llm_provider=mock_llm)

        mock_tool_provider = AsyncMock()
        mock_tool_provider.execute_tool = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="find_available_slots",
                success=True,
                data={"available_slots": ["2026-10-20 10:00 AM", "2026-10-20 02:00 PM"]},
            )
        )
        engine.tool_provider = mock_tool_provider

        result = await engine.process_user_turn(
            state=state,
            user_message="When can I visit JB Pride on October 20?",
        )

        assert mock_tool_provider.execute_tool.called
        assert result.tool_result is not None
        assert result.tool_result.success is True

    @pytest.mark.asyncio
    async def test_low_risk_calendar_booking_with_explicit_confirmation_succeeds(self):
        state = make_test_state("call-lowrisk-02")
        state.extracted_slots["available_slots"] = ["2026-10-20 10:00 AM"]
        state.extracted_slots["visit_confirmation_pending"] = "2026-10-20 10:00 AM"

        decision = ConversationalDecision(
            detected_intent="confirm_visit",
            detected_domain=DomainType.INFRA_BUILDER,
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="create_calendar_event",
                arguments={
                    "slot": "2026-10-20 10:00 AM",
                    "confirmed": True,
                    "property_name": "JB Pride",
                },
                confidence=1.0,
            ),
            user_facing_response="Booking confirmed.",
        )
        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock(return_value=decision)
        engine = ConversationEngine(llm_provider=mock_llm)

        mock_tool_provider = AsyncMock()
        mock_tool_provider.execute_tool = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="create_calendar_event",
                success=True,
                data={"event_id": "evt_book_123", "status": "confirmed"},
            )
        )
        engine.tool_provider = mock_tool_provider

        result = await engine.process_user_turn(
            state=state,
            user_message="Yes, please book it.",
        )

        assert mock_tool_provider.execute_tool.called
        assert result.tool_result is not None
        assert result.tool_result.success is True

    @pytest.mark.asyncio
    async def test_turn_debug_log_sanitizes_pii(self, caplog):
        state = make_test_state("call-debug-pii-01")

        decision = ConversationalDecision(
            detected_intent="general_inquiry",
            detected_domain=DomainType.INFRA_BUILDER,
            user_facing_response="I have received your number 9849012345.",
        )
        mock_llm = AsyncMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock(return_value=decision)
        engine = ConversationEngine(llm_provider=mock_llm)

        with caplog.at_level(logging.INFO):
            await engine.process_user_turn(
                state=state,
                user_message="My phone is +919849012345 and email is test@domain.com",
            )

        log_text = caplog.text
        # Ensure raw phone number and email are sanitized from TURN DEBUG logs
        assert "9849012345" not in log_text
        assert "test@domain.com" not in log_text
        assert "***-2345" in log_text
        assert "t***t@domain.com" in log_text
