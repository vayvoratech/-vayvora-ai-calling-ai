"""Comprehensive test suite for Phase 3D: WebSocket Voice Session Persistence.

Verifies:
1. Reliable persistence of finalized CallSummary upon session termination (normal disconnect or unexpected transport closure).
2. Chronological persistence of interaction dialogue turns in call_interactions distinguishing speaker ("caller" vs "agent").
3. Strict lead preservation: Missing or invalid caller numbers never create leads.
4. Idempotent cleanup: Repeated disconnect/cleanup calls prevent duplicate database writes.
5. Error and timeout resilience: Repository/PostgreSQL failures never break audio/TTS cleanup.
6. Full WebSocket media stream lifecycle persistence integration.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_voice_session_manager, get_conversation_engine, get_state_manager
from src.core.engine import ConversationEngine
from src.core.types import CallDirection, CallMetadata, ConversationStage, DialogueTurn, DomainType, TurnRole
from src.database.connection import check_postgres_health, get_db_pool
from src.database.repository import PostgresRepository
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.telephony.models import CallSummary
from src.telephony.repository import MockCallSessionRepository, PostgresCallSessionRepository
from src.ui.app import app
from src.voice.session_manager import VoiceSessionManager


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def mock_engine() -> ConversationEngine:
    engine = MagicMock(spec=ConversationEngine)
    engine.process_user_turn = AsyncMock()
    return engine


@pytest.fixture
def state_manager() -> ConversationStateManager:
    return ConversationStateManager()


@pytest.fixture
def mock_call_repo() -> MockCallSessionRepository:
    return MockCallSessionRepository()


@pytest.fixture
def voice_manager(mock_engine, state_manager, mock_call_repo) -> VoiceSessionManager:
    return VoiceSessionManager(
        engine=mock_engine,
        state_manager=state_manager,
        stt_service=MagicMock(),
        tts_service=MagicMock(),
        call_repository=mock_call_repo,
    )


# =============================================================================
# 1. Session Persistence & Summary Tests
# =============================================================================

class TestVoiceSessionPersistenceLifecycle:
    """Validate CallSummary and interaction turns persistence upon session completion."""

    @pytest.mark.asyncio
    async def test_normal_disconnect_persists_summary_and_interactions(
        self, voice_manager: VoiceSessionManager, state_manager: ConversationStateManager, mock_call_repo: MockCallSessionRepository
    ):
        """A normal disconnect finalizes the session, persists CallSummary, and records all turns."""
        session_id = "test_normal_disconnect_001"
        caller_phone = "+919988776655"
        caller_name = "Vikram Rao"

        profile = CallerProfile(phone=caller_phone, name=caller_name)
        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.INBOUND,
            caller_profile=profile,
        )

        # Simulate dialogue turns recorded during conversation
        state.record_turn(TurnRole.CALLER, "Hello, can you tell me about Pristine City?")
        state.record_turn(TurnRole.AGENT, "Pristine City features premium plots located in North Bangalore.")
        state.record_turn(TurnRole.CALLER, "What is the price per square yard?")
        state.record_turn(TurnRole.AGENT, "The plots start from 4,500 rupees per square yard.")
        state.current_domain = DomainType.INFRA_BUILDER

        # Simulate active session metadata
        voice_manager._active_sessions[session_id] = {
            "started_at": time.time() - 45.0,
            "state": state,
            "websocket": MagicMock(),
            "tts": MagicMock(),
        }

        # Finalize with normal client disconnect reason
        summary = await voice_manager.finalize_session(session_id=session_id, reason="client_hangup")

        assert summary is not None
        assert summary.session_id == session_id
        assert summary.direction == "inbound"
        assert summary.domain == "infra_builder"
        assert summary.termination_reason == "client_hangup"
        assert summary.duration >= 44.0
        assert summary.caller.get("phone") == caller_phone
        assert summary.caller.get("name") == caller_name

        # Verify summary was stored in repository
        persisted_summary = await mock_call_repo.get_summary(state.metadata.call_id)
        assert persisted_summary is not None
        assert persisted_summary.session_id == session_id

        # Verify interactions were stored in chronological order with correct speakers
        interactions = await mock_call_repo.get_interactions(session_id)
        assert len(interactions) == 4
        assert interactions[0]["speaker"] == "caller"
        assert interactions[0]["text"] == "Hello, can you tell me about Pristine City?"
        assert interactions[1]["speaker"] == "agent"
        assert interactions[1]["text"] == "Pristine City features premium plots located in North Bangalore."
        assert interactions[2]["speaker"] == "caller"
        assert interactions[2]["text"] == "What is the price per square yard?"
        assert interactions[3]["speaker"] == "agent"
        assert interactions[3]["text"] == "The plots start from 4,500 rupees per square yard."

    @pytest.mark.asyncio
    async def test_unexpected_transport_closure_persists_summary(
        self, voice_manager: VoiceSessionManager, state_manager: ConversationStateManager, mock_call_repo: MockCallSessionRepository
    ):
        """Unexpected WebSocket drop records termination reason client_disconnected."""
        session_id = "test_unexpected_drop_002"
        profile = CallerProfile(phone="+919876543210", name="Pawan")
        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.INBOUND,
            caller_profile=profile,
        )
        state.record_turn(TurnRole.CALLER, "Hello?")

        summary = await voice_manager.finalize_session(session_id=session_id, reason="client_disconnected")

        assert summary is not None
        assert summary.termination_reason == "client_disconnected"
        assert not state.conversation_active

        interactions = await mock_call_repo.get_interactions(session_id)
        assert len(interactions) == 1
        assert interactions[0]["speaker"] == "caller"
        assert interactions[0]["text"] == "Hello?"

    @pytest.mark.asyncio
    async def test_chronological_turn_order_speaker_distinction(
        self, voice_manager: VoiceSessionManager, state_manager: ConversationStateManager, mock_call_repo: MockCallSessionRepository
    ):
        """Turns must strictly preserve chronological order and distinguish caller vs agent."""
        session_id = "test_turn_order_003"
        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.OUTBOUND,
            caller_profile=CallerProfile(phone="+919876543210", name="Kiran"),
        )

        turns_data = [
            (TurnRole.AGENT, "Hello Kiran, this is Infra Builder Real Estate."),
            (TurnRole.CALLER, "Hi, I am interested in Serene City."),
            (TurnRole.AGENT, "Serene City offers gated community villa plots near the international airport."),
            (TurnRole.CALLER, "Sounds good, can I schedule a site visit?"),
            (TurnRole.AGENT, "Certainly! What day works best for your site visit?"),
        ]

        for role, text in turns_data:
            state.record_turn(role, text)

        summary = await voice_manager.finalize_session(session_id=session_id, reason="completed")
        assert summary is not None

        interactions = await mock_call_repo.get_interactions(session_id)
        assert len(interactions) == 5

        expected_speakers = ["agent", "caller", "agent", "caller", "agent"]
        for idx, (expected_speaker, (role, text)) in enumerate(zip(expected_speakers, turns_data)):
            assert interactions[idx]["speaker"] == expected_speaker
            assert interactions[idx]["text"] == text


# =============================================================================
# 2. Strict Lead Policy (Missing / Invalid Caller Number)
# =============================================================================

class TestStrictLeadPersistencePolicy:
    """Validate that leads are NEVER created when caller phone number is missing or invalid."""

    @pytest.mark.asyncio
    async def test_missing_phone_does_not_upsert_lead(
        self, mock_engine, state_manager
    ):
        """When phone is None, CallSummary is persisted, but lead is never upserted."""
        mock_pg_repo = MagicMock(spec=PostgresRepository)
        mock_pg_repo.upsert_lead = AsyncMock()
        mock_pg_repo.save_call_interaction = AsyncMock(return_value=1)

        repo = PostgresCallSessionRepository(postgres_repo=mock_pg_repo)
        manager = VoiceSessionManager(
            engine=mock_engine,
            state_manager=state_manager,
            stt_service=MagicMock(),
            tts_service=MagicMock(),
            call_repository=repo,
        )

        session_id = "test_no_phone_004"
        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.INBOUND,
            caller_profile=CallerProfile(phone=None, name="Anonymous Caller"),
        )
        state.record_turn(TurnRole.CALLER, "I have a question.")

        summary = await manager.finalize_session(session_id=session_id, reason="completed")
        assert summary is not None

        # upsert_lead MUST NOT be called because phone is missing
        mock_pg_repo.upsert_lead.assert_not_called()

        # save_call_interaction SHOULD be called for dialogue turns and summary
        assert mock_pg_repo.save_call_interaction.call_count >= 1

    @pytest.mark.asyncio
    async def test_invalid_placeholder_phone_does_not_upsert_lead(
        self, mock_engine, state_manager
    ):
        """Prohibited placeholder numbers (e.g. 'unknown', dummy numbers) do not upsert leads."""
        invalid_numbers = ["unknown", "none", "12345", "0000000000", "default caller", "   "]

        for invalid_phone in invalid_numbers:
            mock_pg_repo = MagicMock(spec=PostgresRepository)
            mock_pg_repo.upsert_lead = AsyncMock()
            mock_pg_repo.save_call_interaction = AsyncMock(return_value=1)

            repo = PostgresCallSessionRepository(postgres_repo=mock_pg_repo)
            manager = VoiceSessionManager(
                engine=mock_engine,
                state_manager=state_manager,
                stt_service=MagicMock(),
                tts_service=MagicMock(),
                call_repository=repo,
            )

            session_id = f"test_invalid_phone_{int(time.time())}_{invalid_phone[:4]}"
            state = state_manager.create_session(
                session_id=session_id,
                domain=DomainType.INFRA_BUILDER,
                direction=CallDirection.INBOUND,
                caller_profile=CallerProfile(phone=invalid_phone, name="Placeholder Caller"),
            )
            state.record_turn(TurnRole.CALLER, "Hello")

            await manager.finalize_session(session_id=session_id, reason="completed")

            mock_pg_repo.upsert_lead.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_phone_upserts_lead_properly(
        self, mock_engine, state_manager
    ):
        """Valid phone number triggers lead upsert with sanitized details."""
        mock_pg_repo = MagicMock(spec=PostgresRepository)
        mock_pg_repo.upsert_lead = AsyncMock(return_value=42)
        mock_pg_repo.save_call_interaction = AsyncMock(return_value=1)

        repo = PostgresCallSessionRepository(postgres_repo=mock_pg_repo)
        manager = VoiceSessionManager(
            engine=mock_engine,
            state_manager=state_manager,
            stt_service=MagicMock(),
            tts_service=MagicMock(),
            call_repository=repo,
        )

        session_id = "test_valid_phone_005"
        valid_phone = "+919845112233"
        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.INBOUND,
            caller_profile=CallerProfile(phone=valid_phone, name="Valid Lead", email="valid@example.com"),
        )
        state.record_turn(TurnRole.CALLER, "I want to invest in Nature Valley.")

        summary = await manager.finalize_session(session_id=session_id, reason="completed")
        assert summary is not None

        # upsert_lead SHOULD be called with clean phone number
        mock_pg_repo.upsert_lead.assert_called_once()
        call_kwargs = mock_pg_repo.upsert_lead.call_args.kwargs
        assert call_kwargs["phone_number"] == valid_phone
        assert call_kwargs["name"] == "Valid Lead"
        assert call_kwargs["email"] == "valid@example.com"


# =============================================================================
# 3. Idempotency & Concurrency Tests
# =============================================================================

class TestVoiceSessionIdempotency:
    """Validate that repeated disconnect / cleanup invocations do not duplicate writes."""

    @pytest.mark.asyncio
    async def test_repeated_finalize_session_is_idempotent(
        self, voice_manager: VoiceSessionManager, state_manager: ConversationStateManager, mock_call_repo: MockCallSessionRepository
    ):
        """Calling finalize_session twice for the same session does not write duplicate interactions or summaries."""
        session_id = "test_idempotent_006"
        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.INBOUND,
            caller_profile=CallerProfile(phone="+919988776655", name="Test Idempotency"),
        )
        state.record_turn(TurnRole.CALLER, "Turn 1")
        state.record_turn(TurnRole.AGENT, "Turn 2")

        # First finalization
        summary_1 = await voice_manager.finalize_session(session_id=session_id, reason="client_hangup")
        assert summary_1 is not None

        # Second finalization (e.g. from subsequent websocket disconnect)
        summary_2 = await voice_manager.finalize_session(session_id=session_id, reason="client_disconnected")
        assert summary_2 is not None
        assert summary_1.session_id == summary_2.session_id

        # Interactions should be exactly 2 (not duplicated to 4)
        interactions = await mock_call_repo.get_interactions(session_id)
        assert len(interactions) == 2


# =============================================================================
# 4. Resilience and Error Isolation Tests
# =============================================================================

class TestVoiceSessionResilience:
    """Validate that database exceptions or timeouts never crash the audio cleanup flow."""

    @pytest.mark.asyncio
    async def test_database_error_does_not_raise_exception(
        self, mock_engine, state_manager
    ):
        """If repository throws an error, finalize_session catches it safely and returns summary."""
        failing_repo = MockCallSessionRepository(force_error=True)
        manager = VoiceSessionManager(
            engine=mock_engine,
            state_manager=state_manager,
            stt_service=MagicMock(),
            tts_service=MagicMock(),
            call_repository=failing_repo,
        )

        session_id = "test_error_resilience_007"
        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.INBOUND,
            caller_profile=CallerProfile(phone="+919876543210", name="Test Resilience"),
        )
        state.record_turn(TurnRole.CALLER, "Hello")

        # Should not raise exception
        summary = await manager.finalize_session(session_id=session_id, reason="completed")
        assert summary is not None

    @pytest.mark.asyncio
    async def test_database_timeout_does_not_hang(
        self, mock_engine, state_manager
    ):
        """If persistence operation takes too long, asyncio.wait_for times out and does not block."""
        async def hanging_persist(*args, **kwargs):
            await asyncio.sleep(10.0)

        slow_repo = MagicMock(spec=PostgresCallSessionRepository)
        slow_repo.save_call_interaction = AsyncMock(side_effect=hanging_persist)
        slow_repo.persist_summary = AsyncMock(side_effect=hanging_persist)

        manager = VoiceSessionManager(
            engine=mock_engine,
            state_manager=state_manager,
            stt_service=MagicMock(),
            tts_service=MagicMock(),
            call_repository=slow_repo,
        )

        session_id = "test_timeout_resilience_008"
        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.INBOUND,
            caller_profile=CallerProfile(phone="+919876543210", name="Test Timeout"),
        )
        state.record_turn(TurnRole.CALLER, "Hello")

        # Should complete within timeout (3.0s) and not hang for 10s
        t0 = time.perf_counter()
        summary = await manager.finalize_session(session_id=session_id, reason="completed")
        elapsed = time.perf_counter() - t0

        assert summary is not None
        assert elapsed < 5.0  # Finished within reasonable time despite 10s sleep


# =============================================================================
# 5. Full WebSocket End-to-End Persistence Integration
# =============================================================================

class TestWebSocketStreamingPersistenceIntegration:
    """Validate end-to-end WebSocket connection disconnect triggering persistence."""

    def test_websocket_client_hangup_triggers_session_finalization(self):
        """WebSocket connection sending hangup event finalizes and cleans up session."""
        client = TestClient(app)
        session_id = f"test_ws_e2e_{int(time.time())}"
        phone = "+919876543210"
        name = "Pawan E2E"

        with client.websocket_connect(
            f"/media-stream?session_id={session_id}&caller_phone={phone}&caller_name={name}"
        ) as ws:
            # Receive initial session start event
            data = ws.receive_json()
            assert data["type"] == "session_started"
            assert data["session_id"] == session_id

            # Send hangup control event
            hangup_msg = {"type": "hangup"}
            ws.send_text(json.dumps(hangup_msg))

        # After WebSocket context closes, session should be finalized
        manager = get_voice_session_manager()
        for _ in range(40):
            if session_id in manager._finalized_summaries:
                break
            time.sleep(0.05)

        assert session_id in manager._finalized_sessions
        assert session_id in manager._finalized_summaries
        summary = manager._finalized_summaries[session_id]
        assert summary.session_id == session_id
        assert summary.caller.get("name") == name


# =============================================================================
# 6. Live PostgreSQL Integration Test
# =============================================================================

class TestLivePostgresVoiceSessionPersistence:
    """Live database test asserting actual writes to call_interactions and leads."""

    @pytest.mark.asyncio
    async def test_live_postgres_persistence_end_to_end(self, mock_engine, state_manager):
        """When PostgreSQL is available on port 5434, turns and leads are saved to database."""
        healthy = await check_postgres_health()
        if not healthy:
            pytest.skip("PostgreSQL instance not reachable on port 5434.")

        postgres_repo = PostgresRepository()
        call_repo = PostgresCallSessionRepository(postgres_repo=postgres_repo)

        manager = VoiceSessionManager(
            engine=mock_engine,
            state_manager=state_manager,
            stt_service=MagicMock(),
            tts_service=MagicMock(),
            call_repository=call_repo,
        )

        unique_id = int(time.time())
        session_id = f"live_sess_{unique_id}"
        test_phone = f"+9198{unique_id % 100000000:08d}"
        test_name = "Live Test Customer"

        state = state_manager.create_session(
            session_id=session_id,
            domain=DomainType.INFRA_BUILDER,
            direction=CallDirection.INBOUND,
            caller_profile=CallerProfile(phone=test_phone, name=test_name, email="customer@test.com"),
        )
        state.record_turn(TurnRole.CALLER, "What is the cost of Nature Valley?")
        state.record_turn(TurnRole.AGENT, "Nature Valley plots are priced from 3,200 per sq yd.")

        summary = await manager.finalize_session(session_id=session_id, reason="completed")
        assert summary is not None

        # Verify turns in database
        interactions = await postgres_repo.get_call_interactions(session_id)
        assert len(interactions) >= 2
        turns_texts = [i["text"] for i in interactions]
        assert "What is the cost of Nature Valley?" in turns_texts
        assert "Nature Valley plots are priced from 3,200 per sq yd." in turns_texts

        # Verify lead in database
        lead = await postgres_repo.get_lead_by_phone(test_phone)
        assert lead is not None
        assert lead["name"] == test_name
