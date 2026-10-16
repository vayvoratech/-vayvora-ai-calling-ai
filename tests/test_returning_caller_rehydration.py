"""Comprehensive tests for Phase 3C: Returning-caller profile restoration from PostgreSQL.

Tests:
1. Known phone number restores correct caller profile.
2. Known caller's name and email are not unnecessarily requested again.
3. Prior preferred project is restored only when supported by stored data.
4. Unknown phone number produces an unknown caller state cleanly.
5. Missing and invalid numbers never trigger identity database lookup.
6. Database errors and timeouts allow inbound state creation to continue gracefully.
7. Restored profile information is available in prompt synthesis.
8. Separate calls receive separate session state while referencing the same persistent caller record.
9. Phone-number matching does not bypass sensitive-action verification.
10. Live database integration test for end-to-end caller rehydration.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest

from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.llm import MockLLMProvider
from src.core.prompts import PromptSynthesizer
from src.core.types import ConversationStage, DomainType
from src.database.connection import get_db_pool
from src.database.repository import PostgresRepository
from src.domains.registry import get_domain_registry
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile
from src.tools.mcp_client import MockToolProvider


class TestReturningCallerProfileRestoration:
    """Verify returning caller profile rehydration from PostgreSQL into ConversationState."""

    @pytest.mark.asyncio
    async def test_known_phone_restores_caller_profile(self):
        """A known phone number restores name, email, caller_type, and initial slots from PostgreSQL."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_lead_by_phone = AsyncMock(
            return_value={
                "id": 101,
                "name": "Kiran Kumar",
                "phone_number": "+919849012345",
                "email": "kiran.kumar@example.com",
                "status": "qualified",
                "interested_in": "Pristine City",
                "metadata": {"source": "inbound_call"},
            }
        )

        manager = ConversationStateManager(postgres_repo=mock_repo)
        state = await manager.create_inbound_state_async(
            call_id="call-known-001",
            caller_phone="+91 98490-12345",
            domain=DomainType.INFRA_BUILDER,
        )

        mock_repo.get_lead_by_phone.assert_called_once_with("+919849012345")
        assert state.caller.name == "Kiran Kumar"
        assert state.caller.phone == "+919849012345"
        assert state.caller.email == "kiran.kumar@example.com"
        assert state.caller.caller_type == "known"
        assert state.caller.is_known() is True
        assert state.caller.known_purpose == "Pristine City"

        # Check slots
        assert state.extracted_slots["caller_name"] == "Kiran Kumar"
        assert state.extracted_slots["email"] == "kiran.kumar@example.com"
        assert state.extracted_slots["preferred_project"] == "Pristine City"
        assert state.extracted_slots["returning_caller"] is True
        assert state.current_interest == "Pristine City"

    @pytest.mark.asyncio
    async def test_known_caller_name_and_email_not_requested_again(self):
        """When caller name and email are already restored, the engine does not prompt for them again."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_lead_by_phone = AsyncMock(
            return_value={
                "id": 102,
                "name": "Pawan Ganesh",
                "phone_number": "+919845112233",
                "email": "pawan@infra.io",
                "status": "inquiry",
                "interested_in": "Nature Valley",
                "metadata": {},
            }
        )

        manager = ConversationStateManager(postgres_repo=mock_repo)
        state = await manager.create_inbound_state_async(
            call_id="call-known-002",
            caller_phone="+919845112233",
            domain=DomainType.INFRA_BUILDER,
        )

        # Process a turn requesting project-team follow-up
        mock_llm = MockLLMProvider()
        mock_tool = MockToolProvider()
        engine = ConversationEngine(llm_provider=mock_llm, tool_provider=mock_tool)

        turn_result = await engine.process_user_turn(
            state, "Please have the project team call me back."
        )

        # The agent should directly record the follow-up request rather than asking "May I have your name?"
        assert "May I have your name" not in turn_result.response_text
        assert state.pending_action is None or state.pending_action != "update_lead"
        assert turn_result.decision.action_proposed is True
        assert turn_result.decision.proposed_action.arguments["name"] == "Pawan Ganesh"

    @pytest.mark.asyncio
    async def test_prior_preferred_project_restored_only_when_stored(self):
        """Preferred project is populated in slots only when supported by database record."""
        mock_repo = MagicMock(spec=PostgresRepository)
        # Lead without interested_in
        mock_repo.get_lead_by_phone = AsyncMock(
            return_value={
                "id": 103,
                "name": "David Miller",
                "phone_number": "+919845998877",
                "email": None,
                "status": "new",
                "interested_in": None,
                "metadata": {},
            }
        )

        manager = ConversationStateManager(postgres_repo=mock_repo)
        state = await manager.create_inbound_state_async(
            call_id="call-known-no-proj",
            caller_phone="+919845998877",
            domain=DomainType.INFRA_BUILDER,
        )

        assert state.caller.name == "David Miller"
        assert state.caller.email is None
        assert "preferred_project" not in state.extracted_slots
        assert state.current_interest is None

    @pytest.mark.asyncio
    async def test_unknown_number_produces_unknown_caller_state(self):
        """When phone is not present in PostgreSQL, session initializes with an unknown caller profile."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_lead_by_phone = AsyncMock(return_value=None)

        manager = ConversationStateManager(postgres_repo=mock_repo)
        state = await manager.create_inbound_state_async(
            call_id="call-unknown-001",
            caller_phone="+919888877777",
            domain=DomainType.INFRA_BUILDER,
        )

        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.phone == "+919888877777"
        assert state.caller.caller_type == "unknown"
        assert state.caller.is_known() is False
        assert "caller_name" not in state.extracted_slots
        assert "returning_caller" not in state.extracted_slots

    @pytest.mark.asyncio
    async def test_missing_and_invalid_numbers_never_trigger_db_lookup(self):
        """Missing, placeholder, or invalid numbers skip DB lookup and initialize as unknown."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_lead_by_phone = AsyncMock()

        manager = ConversationStateManager(postgres_repo=mock_repo)

        invalid_phones = [
            None,
            "",
            "   ",
            "default caller",
            "john doe",
            "0000000000",
            "1234567890",
            "abc",
        ]

        for idx, phone in enumerate(invalid_phones):
            state = await manager.create_inbound_state_async(
                call_id=f"call-inv-{idx}",
                caller_phone=phone,
                domain=DomainType.INFRA_BUILDER,
            )
            assert state.caller.is_known() is False
            assert state.caller.name is None

        mock_repo.get_lead_by_phone.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_timeout_and_error_allow_inbound_state_creation_to_continue(self):
        """Database timeout or connection errors fail closed gracefully to anonymous caller."""
        # 1. Timeout scenario
        timeout_repo = MagicMock(spec=PostgresRepository)
        timeout_repo.get_lead_by_phone = AsyncMock(side_effect=asyncio.TimeoutError())

        manager1 = ConversationStateManager(postgres_repo=timeout_repo)
        state1 = await manager1.create_inbound_state_async(
            call_id="call-timeout-001",
            caller_phone="+919849012345",
            lookup_timeout=0.01,
        )
        assert state1.caller.is_known() is False
        assert state1.caller.name is None
        assert state1.conversation_active is True

        # 2. Connection error scenario
        failing_repo = MagicMock(spec=PostgresRepository)
        failing_repo.get_lead_by_phone = AsyncMock(side_effect=ConnectionError("Database offline"))

        manager2 = ConversationStateManager(postgres_repo=failing_repo)
        state2 = await manager2.create_inbound_state_async(
            call_id="call-offline-001",
            caller_phone="+919849012345",
        )
        assert state2.caller.is_known() is False
        assert state2.caller.name is None
        assert state2.conversation_active is True

    def test_restored_information_available_in_prompt_synthesis(self):
        """Restored caller details appear in system instruction session context."""
        caller = CallerProfile(
            name="Siddharth Rao",
            phone="+919849012345",
            email="siddharth@example.com",
            caller_type="known",
            known_purpose="Pristine City Villa",
        )
        manager = ConversationStateManager(postgres_repo=None)
        state = manager.create_inbound_state(
            call_id="call-prompt-test",
            caller_phone="+919849012345",
            domain=DomainType.INFRA_BUILDER,
            caller_profile=caller,
        )

        domain_config = get_domain_registry().get_or_raise(DomainType.INFRA_BUILDER)
        synthesizer = PromptSynthesizer()
        system_instruction = synthesizer.build_system_instruction(state, domain_config)

        assert "Name: Siddharth Rao" in system_instruction
        assert "Email: siddharth@example.com" in system_instruction
        assert "caller_name: Siddharth Rao" in system_instruction
        assert "preferred_project: Pristine City Villa" in system_instruction

    @pytest.mark.asyncio
    async def test_separate_calls_receive_separate_session_state(self):
        """Two calls from the same phone number receive independent ConversationState instances."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_lead_by_phone = AsyncMock(
            return_value={
                "id": 105,
                "name": "Ananya Sharma",
                "phone_number": "+919849099887",
                "email": "ananya@example.com",
                "status": "qualified",
                "interested_in": "JB Pride",
                "metadata": {},
            }
        )

        manager = ConversationStateManager(postgres_repo=mock_repo)

        # Call 1
        state1 = await manager.create_inbound_state_async(
            call_id="call-sess-001",
            caller_phone="+919849099887",
            domain=DomainType.INFRA_BUILDER,
        )
        # Call 2
        state2 = await manager.create_inbound_state_async(
            call_id="call-sess-002",
            caller_phone="+919849099887",
            domain=DomainType.INFRA_BUILDER,
        )

        assert state1.metadata.call_id == "call-sess-001"
        assert state2.metadata.call_id == "call-sess-002"
        assert state1 is not state2

        # Modifying state1 does not affect state2
        state1.update_slot("budget", "1.5 Crore")
        assert "budget" in state1.extracted_slots
        assert "budget" not in state2.extracted_slots

        # But both share the restored caller identity
        assert state1.caller.name == "Ananya Sharma"
        assert state2.caller.name == "Ananya Sharma"

    @pytest.mark.asyncio
    async def test_phone_number_matching_does_not_bypass_sensitive_action_verification(self):
        """Phone matching marks returning_caller_verified_sensitive as False to prevent unauthorized actions."""
        mock_repo = MagicMock(spec=PostgresRepository)
        mock_repo.get_lead_by_phone = AsyncMock(
            return_value={
                "id": 106,
                "name": "Verified Identity",
                "phone_number": "+919849055443",
                "email": "secure@example.com",
                "status": "active",
                "interested_in": "Serene City",
                "metadata": {},
            }
        )

        manager = ConversationStateManager(postgres_repo=mock_repo)
        state = await manager.create_inbound_state_async(
            call_id="call-sec-001",
            caller_phone="+919849055443",
            domain=DomainType.INFRA_BUILDER,
        )

        # Guard check
        assert state.extracted_slots["returning_caller"] is True
        assert state.extracted_slots["returning_caller_verified_sensitive"] is False

    @pytest.mark.asyncio
    async def test_live_database_integration_caller_rehydration(self):
        """End-to-end integration test with live PostgreSQL: insert lead, rehydrate inbound state."""
        repo = PostgresRepository()
        ts = int(datetime.now(timezone.utc).timestamp())
        phone = f"+1555{ts % 10000000:07d}"

        # 1. Upsert a lead in PostgreSQL
        lead_id = await repo.upsert_lead(
            name="Live Integration Caller",
            phone_number=phone,
            email=f"live_{ts}@infra.com",
            status="interested",
            interested_in="Pristine City",
            metadata={"notes": "Lead created for rehydration verification"},
        )
        assert lead_id is not None

        try:
            # 2. Inbound call arrives with that phone number
            manager = ConversationStateManager(postgres_repo=repo)
            state = await manager.create_inbound_state_async(
                call_id=f"live-call-{ts}",
                caller_phone=phone,
                domain=DomainType.INFRA_BUILDER,
            )

            # 3. Verify restored fields
            assert state.caller.name == "Live Integration Caller"
            assert state.caller.email == f"live_{ts}@infra.com"
            assert state.caller.phone == phone
            assert state.caller.caller_type == "known"
            assert state.caller.is_known() is True
            assert state.extracted_slots["caller_name"] == "Live Integration Caller"
            assert state.extracted_slots["email"] == f"live_{ts}@infra.com"
            assert state.extracted_slots["preferred_project"] == "Pristine City"
            assert state.extracted_slots["returning_caller"] is True
        finally:
            # Clean up test row
            pool = get_db_pool()
            p = await pool.get_pool()
            async with p.acquire() as conn:
                await conn.execute("DELETE FROM leads WHERE phone_number = $1;", phone)
