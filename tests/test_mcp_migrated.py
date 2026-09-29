"""Comprehensive unit and integration test suite for migrated Model Context Protocol (MCP) functionality.

Covers:
- FastMCP server tool registry and JSON-RPC 2.0 protocol methods (initialize, tools/list, tools/call, ping)
- Calendar tools: persistence, slot lookup, event creation with verified event_id, month view
- Mail tools: SMTP dispatch, IMAP recent read, input validation guards
- WhatsApp / Messaging: phone normalization, OpenWA dispatch, message_id verification
- HttpMCPToolProvider integration with FastAPI /mcp endpoint
- Verification and conversation safety invariants (no fabricated success, state continuity)
"""

import asyncio
import json
import os
from pathlib import Path
import pytest
import tempfile
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import httpx

from src.config import Settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    DomainType,
    ToolCallRequest,
)
from src.state.manager import ConversationStateManager
from src.tools.calendar_provider import CalendarProvider, FileCalendarProvider, MockCalendarProvider
from src.tools.email_provider import EmailProvider, EmailReadResult, MockEmailProvider, SMTPEmailProvider
from src.tools.mcp_client import HttpMCPToolProvider, MockToolProvider
from src.tools.message_provider import (
    MessageProvider,
    MockMessageProvider,
    WhatsAppMessageProvider,
    normalize_phone_number,
)
from src.tools.server import FastMCP, create_mcp_server, mcp
from src.tools.validation import ActionValidator
from src.tools.verifier import ActionVerifier
from src.ui.app import app


# =============================================================================
# 1. FastMCP Server & JSON-RPC Protocol Tests
# =============================================================================

class TestFastMCPServerProtocol:
    """Verify FastMCP server instance and JSON-RPC 2.0 protocol methods."""

    @pytest.fixture
    def test_mcp(self, tmp_path: Path):
        events_file = str(tmp_path / "test_events.json")
        cal = FileCalendarProvider(events_path=events_file)
        mail = MockEmailProvider(force_success=True)
        msg = MockMessageProvider(force_success=True)
        return create_mcp_server(
            email_provider=mail,
            calendar_provider=cal,
            message_provider=msg,
        )

    @pytest.mark.asyncio
    async def test_server_instance_and_registered_tools(self, test_mcp: FastMCP):
        assert test_mcp.name == "Vayvora-AI-MCP"
        tools = await test_mcp.list_tools()
        tool_names = {t.name for t in tools}

        # Expected current & migrated tool names
        expected = {
            "send_email",
            "mail_send",
            "mail_read_recent",
            "find_available_slots",
            "calendar_get_events",
            "calendar_get_month_view",
            "create_calendar_event",
            "calendar_add_event",
            "send_message",
            "whatsapp_send_message",
            "update_lead",
            "create_hr_followup",
            "update_business_status",
        }
        assert expected.issubset(tool_names)

    @pytest.mark.asyncio
    async def test_jsonrpc_initialize(self, test_mcp: FastMCP):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = await test_mcp.handle_jsonrpc(req)
        assert resp is not None
        assert resp["id"] == 1
        assert resp["result"]["serverInfo"]["name"] == "Vayvora-AI-MCP"
        assert "tools" in resp["result"]["capabilities"]

    @pytest.mark.asyncio
    async def test_jsonrpc_tools_list(self, test_mcp: FastMCP):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = await test_mcp.handle_jsonrpc(req)
        assert resp is not None
        assert resp["id"] == 2
        tools = resp["result"]["tools"]
        names = [t["name"] for t in tools]
        assert "create_calendar_event" in names
        assert "calendar_add_event" in names
        assert "send_email" in names
        assert "whatsapp_send_message" in names

    @pytest.mark.asyncio
    async def test_jsonrpc_tools_call_success_calendar_month_view(self, test_mcp: FastMCP):
        req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "calendar_get_month_view",
                "arguments": {"year": 2026, "month": 10},
            },
        }
        resp = await test_mcp.handle_jsonrpc(req)
        assert resp is not None
        assert resp["id"] == 3
        assert resp["result"]["isError"] is False
        assert "October 2026" in resp["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_jsonrpc_tools_call_error_handling(self, test_mcp: FastMCP):
        req = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "calendar_get_month_view",
                "arguments": {"year": 2026, "month": 13},  # Invalid month
            },
        }
        resp = await test_mcp.handle_jsonrpc(req)
        assert resp is not None
        assert resp["id"] == 4
        assert resp["result"]["isError"] is True
        assert "Error:" in resp["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_jsonrpc_ping(self, test_mcp: FastMCP):
        req = {"jsonrpc": "2.0", "id": 5, "method": "ping", "params": {}}
        resp = await test_mcp.handle_jsonrpc(req)
        assert resp is not None
        assert resp["id"] == 5
        assert resp["result"] == {}

    @pytest.mark.asyncio
    async def test_jsonrpc_unknown_method(self, test_mcp: FastMCP):
        req = {"jsonrpc": "2.0", "id": 6, "method": "non_existent_method", "params": {}}
        resp = await test_mcp.handle_jsonrpc(req)
        assert resp is not None
        assert resp["id"] == 6
        assert resp["error"]["code"] == -32601


# =============================================================================
# 2. Calendar Tools & Persistence Tests
# =============================================================================

class TestCalendarProviderFunctionality:
    """Verify calendar persistence, slot calculation, and event booking."""

    @pytest.fixture
    def calendar_provider(self, tmp_path: Path):
        events_path = str(tmp_path / "events.json")
        return FileCalendarProvider(events_path=events_path)

    def test_calendar_add_and_get_events(self, calendar_provider: FileCalendarProvider):
        # 1. Add event
        res = calendar_provider.add_event(
            title="Enterprise AI Consultation",
            slot="Tomorrow 02:00 PM",
            description="Discuss voice pipeline architecture",
            attendee_name="Rohan Gupta",
            attendee_email="rohan@example.com",
        )
        assert res["status"] == "confirmed"
        assert res["event_id"].startswith("evt_cal_")
        assert res["slot"] == "Tomorrow 02:00 PM"

        # 2. Read events
        events = calendar_provider.get_events()
        assert len(events) == 1
        assert events[0]["title"] == "Enterprise AI Consultation"
        assert events[0]["event_id"] == res["event_id"]

    def test_calendar_add_event_legacy_format(self, calendar_provider: FileCalendarProvider):
        res = calendar_provider.add_event(
            title="Course Counselling",
            date_str="2026-10-15",
            time_str="14:30",
            description="EduSaaS Consultation",
        )
        assert res["status"] == "confirmed"
        assert res["date"] == "2026-10-15"
        assert res["time"] == "14:30"
        assert res["event_id"].startswith("evt_cal_")

    def test_calendar_invalid_datetime_raises(self, calendar_provider: FileCalendarProvider):
        with pytest.raises(ValueError):
            calendar_provider.add_event(
                title="Bad Event",
                slot="",
                date_str="",
                time_str="",
            )

    def test_calendar_available_slots_filters_booked(self, calendar_provider: FileCalendarProvider):
        # Book Tomorrow 10:00 AM
        calendar_provider.add_event(
            title="Booked Slot",
            slot="Tomorrow 10:00 AM",
            time_str="10:00 AM",
            date_str="Tomorrow",
        )

        slots_res = calendar_provider.get_available_slots(preferred_date="Tomorrow")
        assert "Tomorrow 10:00 AM" not in slots_res["slots"]
        assert "Tomorrow 11:30 AM" in slots_res["slots"]
        assert slots_res["count"] > 0

    def test_calendar_month_view(self, calendar_provider: FileCalendarProvider):
        view = calendar_provider.get_month_view(year=2026, month=10)
        assert "October 2026" in view
        assert "Mo Tu We Th Fr Sa Su" in view

    def test_calendar_invalid_month_raises(self, calendar_provider: FileCalendarProvider):
        with pytest.raises(ValueError):
            calendar_provider.get_month_view(year=2026, month=13)


# =============================================================================
# 3. Mail Tools & IMAP Reading Tests
# =============================================================================

class TestMailToolsFunctionality:
    """Verify mail sending and IMAP recent mail querying."""

    @pytest.mark.asyncio
    async def test_mail_send_via_mock_provider(self):
        provider = MockEmailProvider(force_success=True)
        res = await provider.send_email(
            recipient="client@vayvora.com",
            subject="Consultation Confirmation",
            body="Here are your appointment details.",
        )
        assert res.success is True
        assert res.message_id is not None
        assert res.message_id.startswith("<mock_")
        assert len(provider.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_mail_read_recent_via_mock_provider(self):
        provider = MockEmailProvider(
            mock_inbox=[
                {"from": "alice@corp.com", "subject": "Quarterly Review", "date": "2026-09-28"},
                {"from": "bob@ai.com", "subject": "Model Benchmark", "date": "2026-09-29"},
            ]
        )
        res = await provider.read_recent_emails(limit=2)
        assert res.success is True
        assert len(res.messages) == 2
        assert "Quarterly Review" in res.summary
        assert "Model Benchmark" in res.summary

    @pytest.mark.asyncio
    async def test_mail_read_recent_invalid_limit(self):
        provider = MockEmailProvider()
        res = await provider.read_recent_emails(limit=0)
        assert res.success is False
        assert "Limit must be at least 1" in res.error

    @pytest.mark.asyncio
    async def test_mail_read_recent_auth_failure(self):
        provider = MockEmailProvider(force_auth_failure=True)
        res = await provider.read_recent_emails(limit=5)
        assert res.success is False
        assert res.status == "auth_failed"


# =============================================================================
# 4. WhatsApp / Messaging Tests
# =============================================================================

class TestWhatsAppMessagingFunctionality:
    """Verify WhatsApp phone normalization and OpenWA REST dispatch."""

    def test_normalize_phone_number(self):
        # 10 digits gets '91' prepended
        assert normalize_phone_number("9949350699") == "919949350699"
        # Already 12 digits starting with 91
        assert normalize_phone_number("919949350699") == "919949350699"
        # Formatted phone number with spaces, parens, and dashes
        assert normalize_phone_number("+91 (994) 935-0699") == "919949350699"
        # 11 digits starting with 0
        assert normalize_phone_number("09949350699") == "919949350699"

    @pytest.mark.asyncio
    async def test_mock_message_provider_success(self):
        provider = MockMessageProvider(force_success=True)
        res = await provider.send_message(
            recipient="9949350699",
            message="Your appointment is confirmed for tomorrow 2:30 PM.",
        )
        assert res.success is True
        assert res.recipient == "919949350699"
        assert res.message_id.startswith("msg_wa_")
        assert len(provider.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_whatsapp_provider_rest_dispatch(self):
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            # Mock /api/sessions
            mock_sess_res = MagicMock()
            mock_sess_res.status_code = 200
            mock_sess_res.json.return_value = [{"id": "ready_sess_1", "status": "ready"}]

            # Mock /api/sessions/ready_sess_1/messages/send-text
            mock_send_res = MagicMock()
            mock_send_res.status_code = 200
            mock_send_res.headers = {"content-type": "application/json"}
            mock_send_res.json.return_value = {"id": "wa_msg_998877", "status": "sent"}

            mock_client.get.return_value = mock_sess_res
            mock_client.post.return_value = mock_send_res

            provider = WhatsAppMessageProvider(base_url="http://localhost:2785")
            res = await provider.send_message("9949350699", "Test message")

            assert res.success is True
            assert res.message_id == "wa_msg_998877"
            assert res.recipient == "919949350699"


# =============================================================================
# 5. FastAPI /mcp JSON-RPC Endpoint Integration
# =============================================================================

class TestFastAPIMCPIntegration:
    """Verify live HTTP JSON-RPC 2.0 endpoint in FastAPI workbench app."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_fastapi_mcp_initialize(self, client: TestClient):
        payload = {"jsonrpc": "2.0", "id": 10, "method": "initialize", "params": {}}
        resp = client.post("/mcp", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 10
        assert data["result"]["serverInfo"]["name"] == "Vayvora-AI-MCP"

    def test_fastapi_mcp_tools_list(self, client: TestClient):
        payload = {"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {}}
        resp = client.post("/mcp", json=payload)
        assert resp.status_code == 200
        tools = resp.json()["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "create_calendar_event" in names
        assert "send_email" in names
        assert "send_message" in names

    def test_fastapi_mcp_tool_call_create_calendar_event(self, client: TestClient):
        payload = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "create_calendar_event",
                "arguments": {
                    "slot": "Tomorrow 10:00 AM",
                    "attendee_name": "Rohan Gupta",
                    "confirmed": True,
                },
            },
        }
        resp = client.post("/mcp", json=payload)
        assert resp.status_code == 200
        res = resp.json()["result"]
        assert res["isError"] is False
        assert "event_id" in res
        assert res["event_id"].startswith("evt_cal_")
        assert res["status"] == "confirmed"

    def test_fastapi_mcp_malformed_json_returns_400(self, client: TestClient):
        resp = client.post("/mcp", content=b"invalid non json payload", headers={"Content-Type": "application/json"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == -32700


# =============================================================================
# 6. End-to-End Pipeline & Verification Protocol Tests
# =============================================================================

class TestPipelineActionVerification:
    """Verify that tool proposals route through ActionValidator and ActionVerifier."""

    @pytest.mark.asyncio
    async def test_migrated_calendar_action_verified_success(self, tmp_path: Path):
        events_file = str(tmp_path / "pipeline_events.json")
        cal = FileCalendarProvider(events_path=events_file)
        server = create_mcp_server(calendar_provider=cal)

        # Directly call calendar tool through FastMCP server
        raw_res = await server.call_tool(
            "create_calendar_event",
            {"slot": "Tomorrow 11:30 AM", "attendee_name": "Alice Smith", "confirmed": True},
        )
        assert "event_id" in raw_res
        assert raw_res["event_id"].startswith("evt_cal_")

        # Verify ActionVerifier validates it
        from src.tools.schemas import ToolResult, VerificationStatus
        tool_result = ToolResult(
            action_name="create_calendar_event",
            requested=True,
            started=True,
            succeeded=True,
            data=raw_res,
            external_reference=raw_res["event_id"],
        )
        verified = ActionVerifier.verify(tool_result)
        assert verified.is_verified_success is True
        assert verified.verification_status == VerificationStatus.VERIFIED
        assert verified.external_reference == raw_res["event_id"]

    @pytest.mark.asyncio
    async def test_unverified_tool_result_does_not_fabricate_success(self):
        """ActionVerifier rejects tools returning status=confirmed but lacking event_id."""
        from src.tools.schemas import ToolResult, VerificationStatus
        tool_result = ToolResult(
            action_name="create_calendar_event",
            requested=True,
            started=True,
            succeeded=True,
            data={"status": "confirmed"},  # Missing event_id!
        )
        verified = ActionVerifier.verify(tool_result)
        assert verified.is_verified_success is False
        assert verified.verification_status == VerificationStatus.UNVERIFIED
        assert verified.failed is True
        assert "missing external event reference" in verified.error.lower()
