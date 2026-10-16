"""Comprehensive test suite for Phase 4B: Secure Appointment Persistence and Cancellation.

Verifies:
1. Appointment creation persists the database record and external event association.
2. A missing phone number never creates a fabricated caller or unsafe appointment association.
3. An unauthorized caller cannot list or delete another caller's appointments (cross-caller ownership check).
4. An unverified returning caller cannot cancel an appointment (Phase 3E authorization gate).
5. Zero, one, and multiple appointment candidates are handled correctly (engine disambiguation).
6. Explicit confirmation is required before deletion (cancellation safety violation check).
7. The exact PostgreSQL appointment row is deleted and verified using DELETE ... RETURNING.
8. Calendar deletion failure leaves the database row intact (multi-store consistency).
9. PostgreSQL deletion failure after calendar deletion is surfaced and recoverable (reconciliation inconsistency logged).
10. Retries and duplicate cancellation requests do not delete a different appointment (idempotency and safety).
11. Legacy events.json records remain safe without phone association.
12. Existing booking behavior preserved with explicit confirmation.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, _is_appointment_cancellation_request, _is_explicit_booking_refusal
from src.core.errors import InvalidToolArgumentsError
from src.core.interfaces import LLMProvider, ToolProvider
from src.core.types import (
    CallDirection,
    CallMetadata,
    ConversationStage,
    DomainType,
    ToolCallRequest,
    ToolExecutionResult,
    TurnRole,
)
from src.database.connection import DatabasePool, get_db_pool
from src.database.repository import PostgresRepository, normalize_lead_phone
from src.state.models import CallerProfile, ConversationState
from src.tools.calendar_provider import CalendarEventRecord, FileCalendarProvider, MockCalendarProvider
from src.tools.mcp_client import MockToolProvider
from src.tools.schemas import ToolResult, ValidatedToolRequest, VerificationStatus
from src.tools.server import create_mcp_server
from src.tools.validation import ActionValidator
from src.tools.verifier import ActionVerifier


class DummyLLM(LLMProvider):
    async def generate_response(self, prompt: str, system_instruction: str = "") -> str:
        return json.dumps({
            "detected_domain": "infra_builder",
            "detected_intent": "general_inquiry",
            "proposed_stage": "information",
            "user_facing_response": "I can help you with that.",
            "knowledge_required": False,
        })

    async def generate_decision(self, prompt: str, system_instruction: str = "") -> ConversationalDecision:
        return ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="general_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="I can help you with that.",
            knowledge_required=False,
        )

    async def stream_response(self, prompt: str, system_instruction: str = ""):
        yield "Response"


# =============================================================================
# 1. Database Repository Unit & Live Integration Tests
# =============================================================================

class TestPostgresAppointmentRepository:
    """Validate appointment CRUD in PostgresRepository."""

    @pytest.mark.asyncio
    async def test_01_appointment_creation_persists_database_and_external_id(self):
        """Appointment creation persists record linked with caller phone and external event ID."""
        repo = PostgresRepository()
        phone = "+919849012345"
        appt = await repo.create_appointment(
            phone_number=phone,
            title="Site Visit - Nature Valley",
            event_date="2026-10-25",
            event_time="11:30 AM",
            slot="2026-10-25 11:30 AM",
            project_name="Nature Valley",
            external_event_id="evt_cal_ext_101",
            attendee_name="Pawan",
            attendee_email="pawan@example.com",
        )
        assert appt is not None
        assert appt["id"] is not None
        assert appt["phone_number"] == phone
        assert appt["external_event_id"] == "evt_cal_ext_101"
        assert appt["project_name"] == "Nature Valley"
        assert appt["status"] == "confirmed"

        # Cleanup
        await repo.delete_appointment(appt["id"], phone)

    @pytest.mark.asyncio
    async def test_02_missing_phone_never_creates_synthetic_appointment(self):
        """A missing or invalid phone number is rejected without creating synthetic records."""
        repo = PostgresRepository()
        res1 = await repo.create_appointment(
            phone_number="",
            title="Invalid Visit",
            event_date="2026-10-25",
            event_time="11:30 AM",
            slot="2026-10-25 11:30 AM",
        )
        assert res1 is None

        res2 = await repo.create_appointment(
            phone_number="unknown",
            title="Invalid Visit",
            event_date="2026-10-25",
            event_time="11:30 AM",
            slot="2026-10-25 11:30 AM",
        )
        assert res2 is None

    @pytest.mark.asyncio
    async def test_03_ownership_check_prevents_unauthorized_deletion(self):
        """A caller cannot delete another caller's appointment even if they provide the appointment ID."""
        repo = PostgresRepository()
        victim_phone = "+919876500001"
        attacker_phone = "+919876500002"

        # Create victim's appointment
        appt = await repo.create_appointment(
            phone_number=victim_phone,
            title="Victim Appointment",
            event_date="2026-10-25",
            event_time="10:00 AM",
            slot="2026-10-25 10:00 AM",
            external_event_id="evt_victim_1",
        )
        assert appt is not None
        appt_id = appt["id"]

        try:
            # Attacker attempts deletion with victim's appointment ID
            deleted = await repo.delete_appointment(appt_id, attacker_phone)
            assert deleted is None, "Cross-caller deletion must return None and delete 0 rows"

            # Verify victim's appointment still exists
            remaining = await repo.get_appointment_by_id(appt_id, victim_phone)
            assert remaining is not None
            assert remaining["id"] == appt_id
        finally:
            # Clean up with authorized owner
            await repo.delete_appointment(appt_id, victim_phone)

    @pytest.mark.asyncio
    async def test_04_delete_appointment_returns_deleted_row_and_cleans_up(self):
        """Authorized deletion deletes row using DELETE ... RETURNING."""
        repo = PostgresRepository()
        phone = "+919876543211"
        appt = await repo.create_appointment(
            phone_number=phone,
            title="Site Visit - JB Pride",
            event_date="2026-10-22",
            event_time="03:00 PM",
            slot="2026-10-22 03:00 PM",
            external_event_id="evt_jb_pride_1",
        )
        appt_id = appt["id"]

        deleted = await repo.delete_appointment(appt_id, phone)
        assert deleted is not None
        assert deleted["id"] == appt_id
        assert deleted["phone_number"] == phone

        # Verify no longer returned
        active_appts = await repo.get_appointments_by_phone(phone)
        assert not any(a["id"] == appt_id for a in active_appts)

    @pytest.mark.asyncio
    async def test_05_reconciliation_inconsistency_recording(self):
        """Divergence between calendar provider and DB writes to appointment_reconciliations."""
        repo = PostgresRepository()
        phone = "+919876543212"
        rec_id = await repo.record_reconciliation_inconsistency(
            appointment_id=999,
            external_event_id="evt_unsynced_1",
            phone_number=phone,
            action="delete_sync_failed",
            details="External calendar deleted event, but DB row deletion failed.",
        )
        assert rec_id is not None
        assert isinstance(rec_id, int)


# =============================================================================
# 2. Calendar Provider Deletion & Multi-Store Synchronization
# =============================================================================

class TestCalendarProviderSynchronization:
    """Validate calendar provider deletion and event association."""

    def test_06_calendar_provider_delete_event(self, tmp_path):
        """FileCalendarProvider deletes event by event_id and remains idempotent."""
        events_file = tmp_path / "test_events.json"
        provider = FileCalendarProvider(events_path=str(events_file))

        res = provider.add_event(
            title="Site Visit",
            slot="Tomorrow 10:00 AM",
            phone_number="+919849012345",
        )
        event_id = res["event_id"]
        assert len(provider.get_events()) == 1

        # Deletion succeeds
        deleted = provider.delete_event(event_id)
        assert deleted is True
        assert len(provider.get_events()) == 0

        # Duplicate deletion returns False (already absent) without crashing
        retry_deleted = provider.delete_event(event_id)
        assert retry_deleted is False

    def test_07_legacy_events_json_backward_compatibility(self, tmp_path):
        """Existing events.json records lacking phone_number deserialize safely and remain intact."""
        events_file = tmp_path / "legacy_events.json"
        legacy_data = [
            {
                "id": "evt_legacy_1",
                "event_id": "evt_legacy_1",
                "title": "Legacy Consultation",
                "date": "Tomorrow",
                "time": "10:00 AM",
                "slot": "Tomorrow 10:00 AM",
                "attendee_name": "Legacy Caller",
                "status": "confirmed",
            }
        ]
        events_file.write_text(json.dumps(legacy_data), encoding="utf-8")

        provider = FileCalendarProvider(events_path=str(events_file))
        events = provider.get_events()
        assert len(events) == 1
        assert events[0]["id"] == "evt_legacy_1"
        # Does not crash when inspected as CalendarEventRecord
        record = CalendarEventRecord(**events[0])
        assert record.phone_number is None
        assert record.lead_id is None


# =============================================================================
# 3. FastMCP Server Appointment Creation & Cancellation Tools
# =============================================================================

class TestFastMCPAppointmentTools:
    """Validate create_calendar_event and cancel_calendar_event tools in server.py."""

    @pytest.mark.asyncio
    async def test_08_create_calendar_event_persists_to_both_stores(self):
        """create_calendar_event writes to both calendar provider and PostgreSQL."""
        mock_cal = MockCalendarProvider()
        repo = PostgresRepository()
        server = create_mcp_server(calendar_provider=mock_cal, postgres_repo=repo)

        phone = "+919876543215"
        res = await server.call_tool(
            "create_calendar_event",
            {
                "slot": "Tomorrow 10:00 AM",
                "title": "Pristine City Site Visit",
                "phone_number": phone,
                "property_name": "Pristine City",
                "confirmed": True,
            },
        )
        assert res["status"] == "confirmed"
        assert res["event_id"] is not None
        assert res["appointment_id"] is not None

        # Verify in mock calendar
        assert len(mock_cal.events) == 1

        # Verify in DB
        db_appt = await repo.get_appointment_by_id(res["appointment_id"], phone)
        assert db_appt is not None
        assert db_appt["external_event_id"] == res["event_id"]

        # Cleanup
        await repo.delete_appointment(res["appointment_id"], phone)

    @pytest.mark.asyncio
    async def test_09_create_calendar_event_failure_rolls_back_calendar_provider(self):
        """If database write fails during appointment creation, calendar provider write is compensated."""
        mock_cal = MockCalendarProvider()
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.normalize_lead_phone = normalize_lead_phone
        # Database raises exception
        mock_repo.create_appointment = AsyncMock(side_effect=RuntimeError("Simulated DB connection drop"))

        server = create_mcp_server(calendar_provider=mock_cal, postgres_repo=mock_repo)

        with pytest.raises(RuntimeError, match="Failed to persist appointment to database"):
            await server.call_tool(
                "create_calendar_event",
                {
                    "slot": "Tomorrow 10:00 AM",
                    "title": "Failed DB Visit",
                    "phone_number": "+919876543216",
                    "confirmed": True,
                },
            )

        # Invariant: mock_cal must not retain the event (it was compensated / deleted)
        assert len(mock_cal.events) == 0

    @pytest.mark.asyncio
    async def test_10_cancel_calendar_event_requires_sensitive_authorization(self):
        """Unverified returning caller is rejected by the sensitive authorization gate."""
        mock_cal = MockCalendarProvider()
        repo = PostgresRepository()
        server = create_mcp_server(calendar_provider=mock_cal, postgres_repo=repo)

        with pytest.raises(PermissionError, match="Cancelling an appointment requires verified caller identity"):
            await server.call_tool(
                "cancel_calendar_event",
                {
                    "appointment_id": 1,
                    "phone_number": "+919876543217",
                    "confirmed": True,
                    "returning_caller": True,
                    "caller_verified_sensitive": False,
                },
            )

    @pytest.mark.asyncio
    async def test_11_cancel_calendar_event_requires_explicit_confirmation(self):
        """cancel_calendar_event without confirmed=True fails closed."""
        mock_cal = MockCalendarProvider()
        repo = PostgresRepository()
        server = create_mcp_server(calendar_provider=mock_cal, postgres_repo=repo)

        with pytest.raises(ValueError, match="Explicit caller confirmation is required"):
            await server.call_tool(
                "cancel_calendar_event",
                {
                    "appointment_id": 1,
                    "phone_number": "+919876543218",
                    "confirmed": False,
                    "caller_verified_sensitive": True,
                },
            )

    @pytest.mark.asyncio
    async def test_12_cancel_calendar_event_atomic_multi_store_deletion(self):
        """Verified cancellation deletes external calendar event and PostgreSQL row atomically."""
        mock_cal = MockCalendarProvider()
        repo = PostgresRepository()
        server = create_mcp_server(calendar_provider=mock_cal, postgres_repo=repo)

        phone = "+919876543219"
        create_res = await server.call_tool(
            "create_calendar_event",
            {
                "slot": "Tomorrow 02:00 PM",
                "title": "Serene City Visit",
                "phone_number": phone,
                "property_name": "Serene City",
                "confirmed": True,
            },
        )
        appt_id = create_res["appointment_id"]
        ext_id = create_res["event_id"]

        assert len(mock_cal.events) == 1
        assert await repo.get_appointment_by_id(appt_id, phone) is not None

        # Execute verified cancellation
        cancel_res = await server.call_tool(
            "cancel_calendar_event",
            {
                "appointment_id": appt_id,
                "phone_number": phone,
                "confirmed": True,
                "caller_verified_sensitive": True,
            },
        )
        assert cancel_res["cancelled"] is True
        assert cancel_res["status"] == "cancelled"

        # Verify both stores are cleaned up
        assert len(mock_cal.events) == 0
        assert await repo.get_appointment_by_id(appt_id, phone) is None

    @pytest.mark.asyncio
    async def test_13_calendar_deletion_failure_leaves_database_row_intact(self):
        """If calendar provider deletion fails, PostgreSQL row is retained."""
        # Calendar provider configured to fail
        failing_cal = MockCalendarProvider(force_failure=True)
        repo = PostgresRepository()

        # Create row in DB
        phone = "+919876543220"
        appt = await repo.create_appointment(
            phone_number=phone,
            title="Safe DB Row",
            event_date="2026-10-30",
            event_time="10:00 AM",
            slot="2026-10-30 10:00 AM",
            external_event_id="evt_fail_cal",
        )
        appt_id = appt["id"]

        server = create_mcp_server(calendar_provider=failing_cal, postgres_repo=repo)

        cancel_res = await server.call_tool(
            "cancel_calendar_event",
            {
                "appointment_id": appt_id,
                "external_event_id": "evt_fail_cal",
                "phone_number": phone,
                "confirmed": True,
                "caller_verified_sensitive": True,
            },
        )
        # Cancel failed, appointment retained
        assert cancel_res["cancelled"] is False
        assert cancel_res["status"] == "failed"

        # Invariant: DB row was NOT deleted!
        still_there = await repo.get_appointment_by_id(appt_id, phone)
        assert still_there is not None

        # Cleanup
        await repo.delete_appointment(appt_id, phone)


# =============================================================================
# 4. Action Verifier Integration
# =============================================================================

class TestActionVerifierCancellation:
    """Validate ActionVerifier rules for cancel_calendar_event."""

    def test_14_action_verifier_validates_cancellation(self):
        """ActionVerifier marks verified cancellation as VERIFIED."""
        res = ToolResult(
            action_name="cancel_calendar_event",
            requested=True,
            started=True,
            succeeded=True,
            external_reference="123",
            data={"appointment_id": 123, "cancelled": True, "status": "cancelled"},
        )
        verified = ActionVerifier.verify(res)
        assert verified.verification_status == VerificationStatus.VERIFIED

    def test_15_action_verifier_rejects_unconfirmed_cancellation(self):
        """ActionVerifier rejects cancellation response missing cancelled confirmation."""
        res = ToolResult(
            action_name="cancel_calendar_event",
            requested=True,
            started=True,
            succeeded=True,
            data={"status": "unknown"},  # Missing cancelled=True and reference
        )
        verified = ActionVerifier.verify(res)
        assert verified.verification_status == VerificationStatus.UNVERIFIED


# =============================================================================
# 5. Conversational Engine Intent Detection, Disambiguation & Confirmation
# =============================================================================

class TestEngineAppointmentCancellation:
    """Validate engine cancellation workflow, target disambiguation, and refusal separation."""

    def test_16_distinguish_booking_refusal_from_cancellation_request(self):
        """'I don't want to book' is a refusal; 'Cancel my appointment' is a cancellation request."""
        assert _is_explicit_booking_refusal("I don't want to book a visit") is True
        assert _is_explicit_booking_refusal("Please do not book any slot") is True
        assert _is_explicit_booking_refusal("Cancel my appointment") is False
        assert _is_explicit_booking_refusal("Please cancel my booking") is False

        assert _is_appointment_cancellation_request("Cancel my appointment") is True
        assert _is_appointment_cancellation_request("Please cancel my site visit") is True
        assert _is_appointment_cancellation_request("I want to delete my booking") is True
        assert _is_appointment_cancellation_request("I don't want to book a visit") is False

    @pytest.mark.asyncio
    async def test_17_cancellation_unverified_caller_rejected_by_gate(self):
        """Unverified caller requesting cancellation is informed of identity requirement."""
        engine = ConversationEngine(llm_provider=DummyLLM())
        state = ConversationState(
            metadata=CallMetadata(call_id="call_unverified_1", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919876543221"),
        )
        state.update_slot("returning_caller", True)
        state.update_slot("returning_caller_verified_sensitive", False)

        result = await engine.process_user_turn(state, "Please cancel my appointment")
        assert "verified caller authentication" in result.response_text.lower()
        assert state.pending_action is None

    @pytest.mark.asyncio
    async def test_18_cancellation_zero_candidates_found(self):
        """Verified caller with no scheduled appointments receives clear not-found explanation."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_appointments_by_phone = AsyncMock(return_value=[])

        engine = ConversationEngine(llm_provider=DummyLLM(), postgres_repo=mock_repo)
        state = ConversationState(
            metadata=CallMetadata(call_id="call_zero_1", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919876543222"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)

        result = await engine.process_user_turn(state, "Cancel my appointment")
        assert "couldn't find any scheduled appointments" in result.response_text.lower()

    @pytest.mark.asyncio
    async def test_19_cancellation_single_candidate_requests_confirmation(self):
        """Single active appointment prompts caller with exact details for confirmation."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_appointments_by_phone = AsyncMock(
            return_value=[
                {
                    "id": 42,
                    "project_name": "Pristine City",
                    "title": "Site Visit",
                    "event_date": "2026-10-25",
                    "event_time": "10:00 AM",
                    "slot": "2026-10-25 10:00 AM",
                }
            ]
        )

        engine = ConversationEngine(llm_provider=DummyLLM(), postgres_repo=mock_repo)
        state = ConversationState(
            metadata=CallMetadata(call_id="call_single_1", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919876543223"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)

        result = await engine.process_user_turn(state, "Cancel my visit")
        assert "Pristine City" in result.response_text
        assert "2026-10-25" in result.response_text
        assert "Are you sure you would like to cancel" in result.response_text
        assert state.pending_action == "cancel_calendar_event"
        assert state.get_slot("cancellation_confirmation_pending_id") == 42

    @pytest.mark.asyncio
    async def test_20_cancellation_multiple_candidates_prompts_disambiguation(self):
        """Multiple active appointments prompt caller to specify which one they mean."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_appointments_by_phone = AsyncMock(
            return_value=[
                {
                    "id": 101,
                    "project_name": "JB Pride",
                    "event_date": "2026-10-22",
                    "event_time": "10:00 AM",
                },
                {
                    "id": 102,
                    "project_name": "Nature Valley",
                    "event_date": "2026-10-26",
                    "event_time": "03:00 PM",
                },
            ]
        )

        engine = ConversationEngine(llm_provider=DummyLLM(), postgres_repo=mock_repo)
        state = ConversationState(
            metadata=CallMetadata(call_id="call_multi_1", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919876543224"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)

        result = await engine.process_user_turn(state, "Cancel my appointment")
        assert "multiple appointments scheduled" in result.response_text.lower()
        assert "JB Pride" in result.response_text
        assert "Nature Valley" in result.response_text
        assert "Which appointment would you like to cancel?" in result.response_text

    @pytest.mark.asyncio
    async def test_21_end_to_end_multi_turn_cancellation_execution(self):
        """Turn 1 prompts confirmation; Turn 2 confirms and dispatches cancellation tool."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_appointments_by_phone = AsyncMock(
            return_value=[
                {
                    "id": 42,
                    "project_name": "Pristine City",
                    "title": "Site Visit",
                    "event_date": "2026-10-25",
                    "event_time": "10:00 AM",
                    "slot": "2026-10-25 10:00 AM",
                }
            ]
        )
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=DummyLLM(), postgres_repo=mock_repo, tool_provider=mock_tools)

        state = ConversationState(
            metadata=CallMetadata(call_id="call_multi_turn_1", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012399"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)

        # Turn 1: Caller requests cancellation
        turn1_result = await engine.process_user_turn(state, "Please cancel my site visit")
        assert "Are you sure you would like to cancel" in turn1_result.response_text
        assert state.pending_action == "cancel_calendar_event"
        assert state.get_slot("cancellation_confirmation_pending_id") == 42
        assert len(mock_tools.dispatched_requests) == 0

        # Turn 2: Caller confirms cancellation
        turn2_result = await engine.process_user_turn(state, "Yes, please cancel it")
        assert "Your appointment has been successfully cancelled" in turn2_result.response_text
        assert state.pending_action is None
        assert state.get_slot("cancellation_confirmation_pending_id") is None
        assert len(mock_tools.dispatched_requests) == 1
        dispatched = mock_tools.dispatched_requests[0]
        assert dispatched.action_name == "cancel_calendar_event"
        assert dispatched.arguments["appointment_id"] == 42
        assert dispatched.arguments["confirmed"] is True
        assert dispatched.arguments["caller_verified_sensitive"] is True

    @pytest.mark.asyncio
    async def test_22_cancellation_negative_confirmation_preserves_appointment(self):
        """Caller saying 'No, keep it' aborts cancellation and does NOT dispatch tool."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_appointments_by_phone = AsyncMock(
            return_value=[
                {
                    "id": 43,
                    "project_name": "Serene City",
                    "title": "Site Visit",
                    "event_date": "2026-10-28",
                    "event_time": "11:00 AM",
                }
            ]
        )
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=DummyLLM(), postgres_repo=mock_repo, tool_provider=mock_tools)

        state = ConversationState(
            metadata=CallMetadata(call_id="call_abort_1", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012398"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)

        # Turn 1: Request
        await engine.process_user_turn(state, "Cancel my appointment")
        assert state.pending_action == "cancel_calendar_event"

        # Turn 2: Abort
        abort_result = await engine.process_user_turn(state, "No, keep it please, don't delete it")
        assert "keep your scheduled appointment intact" in abort_result.response_text
        assert state.pending_action is None
        assert state.get_slot("cancellation_confirmation_pending_id") is None
        assert len(mock_tools.dispatched_requests) == 0

    @pytest.mark.asyncio
    async def test_23_direct_unconfirmed_llm_cancellation_proposal_rejected(self):
        """Engine rejects direct unconfirmed LLM cancellation proposal without caller confirmation."""
        class MaliciousLLM(LLMProvider):
            async def generate_response(self, prompt: str, system_instruction: str = "") -> str:
                return ""

            async def generate_decision(self, prompt: str, system_instruction: str = "") -> ConversationalDecision:
                return ConversationalDecision(
                    detected_domain=DomainType.INFRA_BUILDER,
                    detected_intent="cancel_appointment",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response="Cancelling your appointment now.",
                    action_proposed=True,
                    proposed_action=ProposedAction(
                        tool_name="cancel_calendar_event",
                        arguments={"appointment_id": 999, "confirmed": False},
                    ),
                )

            async def stream_response(self, prompt: str, system_instruction: str = ""):
                yield ""

        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=MaliciousLLM(), tool_provider=mock_tools)
        state = ConversationState(
            metadata=CallMetadata(call_id="call_bypass_1", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012397"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)

        result = await engine.process_user_turn(state, "Hello")
        assert len(mock_tools.dispatched_requests) == 0
        assert "safety" in result.response_text.lower() or "confirm" in result.response_text.lower()

    def test_24_action_validator_enforces_cancellation_schema(self):
        """ActionValidator strictly validates cancellation parameters."""
        # Valid confirmed request
        valid_req = ValidatedToolRequest(
            action_name="cancel_calendar_event",
            arguments={"appointment_id": 10, "confirmed": True, "phone_number": "+919849012345"},
            session_id="sess_val_1",
            idempotency_key="key_1",
            domain=DomainType.INFRA_BUILDER,
        )
        ActionValidator.validate_action(valid_req)

        # Unconfirmed request
        unconfirmed_req = ValidatedToolRequest(
            action_name="cancel_calendar_event",
            arguments={"appointment_id": 10, "confirmed": False},
            session_id="sess_val_2",
            idempotency_key="key_2",
            domain=DomainType.INFRA_BUILDER,
        )
        with pytest.raises(InvalidToolArgumentsError, match="explicitly confirm"):
            ActionValidator.validate_action(unconfirmed_req)

        # Missing appointment ID
        missing_id_req = ValidatedToolRequest(
            action_name="cancel_calendar_event",
            arguments={"confirmed": True},
            session_id="sess_val_3",
            idempotency_key="key_3",
            domain=DomainType.INFRA_BUILDER,
        )
        with pytest.raises(InvalidToolArgumentsError, match="appointment ID or external event ID is required"):
            ActionValidator.validate_action(missing_id_req)

    @pytest.mark.asyncio
    async def test_25_missing_verified_sensitive_fails_closed(self):
        """Caller missing verified_sensitive fails closed and is never authorized."""
        engine = ConversationEngine(llm_provider=DummyLLM())
        state = ConversationState(
            metadata=CallMetadata(call_id="call_fail_closed_1", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012396"),
        )
        # Ensure no sensitive authorization slot exists
        assert state.get_slot("returning_caller_verified_sensitive") is None

        result = await engine.process_user_turn(state, "Cancel my site visit")
        assert "verified caller authentication" in result.response_text.lower()
        assert state.pending_action is None

    @pytest.mark.asyncio
    async def test_26_authorized_affirmative_confirmation_executes_once_zero_llm_calls(self):
        """Verified caller affirmative cancellation executes tool once with ZERO cloud LLM calls."""
        mock_tools = MockToolProvider()
        mock_llm = MagicMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock()
        mock_llm.generate_response = AsyncMock()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)

        state = ConversationState(
            metadata=CallMetadata(call_id="call_phase5b_26", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012350"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)
        state.set_pending_action("cancel_calendar_event")
        state.update_slot("cancellation_confirmation_pending_id", 50)
        state.update_slot("cancellation_confirmation_pending_details", "Nature Valley on 2026-10-25 at 11:30 AM")
        state.set_pending_question("Are you sure you would like to cancel your appointment?")

        # Caller confirms cancellation
        result = await engine.process_user_turn(state, "Yes, please cancel it")

        # Zero cloud LLM calls made!
        assert mock_llm.generate_decision.call_count == 0
        assert mock_llm.generate_response.call_count == 0

        # Exact cancellation tool executed once
        assert len(mock_tools.dispatched_requests) == 1
        dispatched = mock_tools.dispatched_requests[0]
        assert dispatched.action_name == "cancel_calendar_event"
        assert dispatched.arguments["appointment_id"] == 50
        assert dispatched.arguments["confirmed"] is True
        assert dispatched.arguments["caller_verified_sensitive"] is True

        # State and response verified
        assert "Your appointment has been successfully cancelled" in result.response_text
        assert state.pending_action is None
        assert state.get_slot("cancellation_confirmation_pending_id") is None
        assert state.get_slot("cancellation_confirmation_pending_details") is None
        assert result.tool_result is not None
        assert result.tool_result.success is True

    @pytest.mark.asyncio
    async def test_27_negative_confirmation_performs_no_deletion(self):
        """Caller negative response aborts cancellation with zero LLM calls and zero tool dispatches."""
        mock_tools = MockToolProvider()
        mock_llm = MagicMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock()
        mock_llm.generate_response = AsyncMock()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)

        state = ConversationState(
            metadata=CallMetadata(call_id="call_phase5b_27", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012351"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)
        state.set_pending_action("cancel_calendar_event")
        state.update_slot("cancellation_confirmation_pending_id", 51)
        state.update_slot("cancellation_confirmation_pending_details", "Pristine City on 2026-10-26 at 10:00 AM")
        state.set_pending_question("Are you sure you would like to cancel your appointment?")

        result = await engine.process_user_turn(state, "No, don't cancel it, keep my appointment please")

        # Zero cloud LLM calls made!
        assert mock_llm.generate_decision.call_count == 0
        assert mock_llm.generate_response.call_count == 0

        # No deletion tool dispatched!
        assert len(mock_tools.dispatched_requests) == 0

        # Appointment preserved, pending state cleared
        assert "keep your scheduled appointment intact" in result.response_text
        assert state.pending_action is None
        assert state.get_slot("cancellation_confirmation_pending_id") is None
        assert state.get_slot("cancellation_confirmation_pending_details") is None

    @pytest.mark.asyncio
    async def test_28_ambiguous_and_qualified_responses_do_not_delete(self):
        """Ambiguous or qualified responses do NOT delete or abort; they keep pending state and clarify."""
        ambiguous_utterances = [
            "Maybe, I'm not sure yet",
            "Wait, what time was it?",
            "Yes, but tell me the date first",
            "Not yet, hold on a sec",
        ]
        for utterance in ambiguous_utterances:
            mock_tools = MockToolProvider()
            mock_llm = MagicMock(spec=LLMProvider)
            mock_llm.generate_decision = AsyncMock()
            mock_llm.generate_response = AsyncMock()
            engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)

            state = ConversationState(
                metadata=CallMetadata(call_id="call_phase5b_28", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
                current_domain=DomainType.INFRA_BUILDER,
                primary_domain=DomainType.INFRA_BUILDER,
                caller=CallerProfile(phone="+919849012352"),
            )
            state.update_slot("returning_caller_verified_sensitive", True)
            state.set_pending_action("cancel_calendar_event")
            state.update_slot("cancellation_confirmation_pending_id", 52)
            state.update_slot("cancellation_confirmation_pending_details", "Serene City on 2026-10-27 at 2:00 PM")
            state.set_pending_question("Are you sure you would like to cancel your appointment?")

            result = await engine.process_user_turn(state, utterance)

            # Zero LLM calls
            assert mock_llm.generate_decision.call_count == 0
            assert mock_llm.generate_response.call_count == 0

            # No tool execution / deletion
            assert len(mock_tools.dispatched_requests) == 0

            # Pending cancellation state MUST BE PRESERVED
            assert state.pending_action == "cancel_calendar_event"
            assert state.get_slot("cancellation_confirmation_pending_id") == 52
            assert state.get_slot("cancellation_confirmation_pending_details") == "Serene City on 2026-10-27 at 2:00 PM"
            assert "Just to be certain, would you like me to cancel" in result.response_text

    @pytest.mark.asyncio
    async def test_29_no_pending_confirmation_means_no_deterministic_cancellation(self):
        """When no cancellation is pending, deterministic cancellation returns None and standard processing occurs."""
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=DummyLLM(), tool_provider=mock_tools)

        state = ConversationState(
            metadata=CallMetadata(call_id="call_phase5b_29", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012353"),
        )
        # Ensure no pending cancellation
        assert state.pending_action is None
        assert state.get_slot("cancellation_confirmation_pending_id") is None

        result = await engine.process_user_turn(state, "Yes, please")
        # Tool was not dispatched
        assert len(mock_tools.dispatched_requests) == 0
        # DummyLLM was called
        assert "I can help you with that." in result.response_text

    @pytest.mark.asyncio
    async def test_30_missing_or_inconsistent_pending_appointment_state_fails_closed(self):
        """Corrupted or missing appointment ID during confirmation fails closed without deleting."""
        test_cases = [
            ("missing_id", None, "+919849012354"),
            ("negative_id", -1, "+919849012354"),
            ("string_corrupted_id", "not_an_id", "+919849012354"),
            ("missing_phone", 54, None),
        ]
        for name, appt_id, phone in test_cases:
            mock_tools = MockToolProvider()
            engine = ConversationEngine(llm_provider=DummyLLM(), tool_provider=mock_tools)

            state = ConversationState(
                metadata=CallMetadata(call_id=f"call_phase5b_30_{name}", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
                current_domain=DomainType.INFRA_BUILDER,
                primary_domain=DomainType.INFRA_BUILDER,
                caller=CallerProfile(phone=phone) if phone else CallerProfile(),
            )
            state.update_slot("returning_caller_verified_sensitive", True)
            state.set_pending_action("cancel_calendar_event")
            if appt_id is not None:
                state.update_slot("cancellation_confirmation_pending_id", appt_id)

            result = await engine.process_user_turn(state, "Yes, please cancel it")

            # Fails closed: no deletion dispatched
            assert len(mock_tools.dispatched_requests) == 0
            assert "unable to verify the appointment details" in result.response_text
            assert state.pending_action is None
            assert state.get_slot("cancellation_confirmation_pending_id") is None

    @pytest.mark.asyncio
    async def test_31_unverified_caller_cannot_execute_cancellation(self):
        """Unverified caller attempting to confirm cancellation is rejected by sensitive-action gate."""
        mock_tools = MockToolProvider()
        engine = ConversationEngine(llm_provider=DummyLLM(), tool_provider=mock_tools)

        state = ConversationState(
            metadata=CallMetadata(call_id="call_phase5b_31", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012355"),
        )
        # Explicitly ensure caller is NOT verified sensitive
        assert not state.get_slot("returning_caller_verified_sensitive")
        state.set_pending_action("cancel_calendar_event")
        state.update_slot("cancellation_confirmation_pending_id", 55)

        result = await engine.process_user_turn(state, "Yes, please cancel it")

        # Zero tool executions
        assert len(mock_tools.dispatched_requests) == 0
        assert "verified caller authentication" in result.response_text.lower()
        assert state.pending_action is None
        assert state.get_slot("cancellation_confirmation_pending_id") is None

    @pytest.mark.asyncio
    async def test_32_calendar_or_postgres_failure_not_reported_as_success(self):
        """When tool execution fails, engine does NOT claim success and safely retains record."""
        class FailingCancelToolProvider(MockToolProvider):
            async def execute_tool(self, request: ToolCallRequest) -> ToolExecutionResult:
                return ToolExecutionResult(
                    tool_name=request.tool_name,
                    success=False,
                    data={},
                    error_message="PostgreSQL connection terminated during transaction",
                )

        engine = ConversationEngine(llm_provider=DummyLLM(), tool_provider=FailingCancelToolProvider())

        state = ConversationState(
            metadata=CallMetadata(call_id="call_phase5b_32", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012356"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)
        state.set_pending_action("cancel_calendar_event")
        state.update_slot("cancellation_confirmation_pending_id", 56)

        result = await engine.process_user_turn(state, "Yes, please cancel it")

        # Crucial invariant: never falsely claim success when tool execution failed
        assert "successfully cancelled" not in result.response_text
        assert "couldn't cancel that appointment due to a system issue" in result.response_text
        assert "retained in our records" in result.response_text
        assert state.pending_action is None

    @pytest.mark.asyncio
    async def test_33_existing_multi_turn_cancellation_and_unrelated_dialogue_preserved(self):
        """When caller asks an unrelated question while cancellation is pending, standard processing preserves inquiry."""
        mock_tools = MockToolProvider()
        mock_llm = MagicMock(spec=LLMProvider)
        mock_llm.generate_decision = AsyncMock(
            return_value=ConversationalDecision(
                detected_domain=DomainType.INFRA_BUILDER,
                detected_intent="property_location_inquiry",
                proposed_stage=ConversationStage.INFORMATION,
                user_facing_response="Pristine City is located near the Financial District.",
                knowledge_required=False,
            )
        )
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tools)

        state = ConversationState(
            metadata=CallMetadata(call_id="call_phase5b_33", direction=CallDirection.INBOUND, primary_domain=DomainType.INFRA_BUILDER),
            current_domain=DomainType.INFRA_BUILDER,
            primary_domain=DomainType.INFRA_BUILDER,
            caller=CallerProfile(phone="+919849012357"),
        )
        state.update_slot("returning_caller_verified_sensitive", True)
        state.set_pending_action("cancel_calendar_event")
        state.update_slot("cancellation_confirmation_pending_id", 57)
        state.update_slot("cancellation_confirmation_pending_details", "Pristine City on 2026-10-29 at 3:00 PM")

        # Caller asks unrelated location question instead of confirming or denying
        result = await engine.process_user_turn(state, "Where is Pristine City located?")

        # Tool was NOT dispatched (no cancellation executed!)
        assert len(mock_tools.dispatched_requests) == 0
        # Decision LLM was called to answer the inquiry
        assert mock_llm.generate_decision.call_count == 1
        assert "Financial District" in result.response_text
