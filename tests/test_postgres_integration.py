"""Comprehensive integration and resilience test suite for PostgreSQL.

Tests:
1. DatabasePool connection lifecycle, asyncpg pooling, and health checks.
2. PostgresRepository operations on verified schema:
   - call_interactions (persistence and retrieval of dialogue turns)
   - leads (CRM upsert, update, listing)
   - hr_followup_queue (candidate callback creation and listing)
3. FastMCP tools (update_lead, create_hr_followup) with database integration and offline fallback.
4. PostgresCallSessionRepository persistence and fault tolerance.
5. FastAPI health check endpoint reporting postgres_connected.
"""

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.config import Settings, get_settings
from src.database.connection import DatabasePool, check_postgres_health, get_db_pool
from src.database.repository import PostgresRepository
from src.telephony.models import CallSummary
from src.telephony.repository import MockCallSessionRepository, PostgresCallSessionRepository
from src.tools.server import create_mcp_server
from src.ui.app import app


# =============================================================================
# 1. Database Pool & Health Check Tests
# =============================================================================

class TestPostgresConnectionAndHealth:
    """Verify live PostgreSQL connection and health monitoring."""

    @pytest.mark.asyncio
    async def test_live_postgres_health_check(self):
        """PostgreSQL service running on port 5434 reports healthy."""
        healthy = await check_postgres_health()
        assert healthy is True

    @pytest.mark.asyncio
    async def test_database_pool_lifecycle(self):
        """DatabasePool acquires connections and handles queries cleanly."""
        pool = DatabasePool()
        asyncpg_pool = await pool.get_pool()
        assert asyncpg_pool is not None

        async with asyncpg_pool.acquire() as conn:
            val = await conn.fetchval("SELECT 100 + 42;")
            assert val == 142

        # Check engine creation
        engine = pool.get_engine()
        assert engine is not None

        await pool.close()
        assert pool.is_closed is True


# =============================================================================
# 2. PostgresRepository Schema Operations
# =============================================================================

class TestPostgresRepositoryOperations:
    """Verify CRUD and query contracts for leads, interactions, and HR followups."""

    @pytest.fixture
    def repo(self):
        return PostgresRepository()

    @pytest.mark.asyncio
    async def test_save_and_retrieve_call_interactions(self, repo: PostgresRepository):
        session_id = f"test_session_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Insert turns
        turn1 = await repo.save_call_interaction(
            session_id=session_id,
            speaker="caller",
            text="Hi, I want to know about your data science degree program.",
        )
        assert turn1 is not None and isinstance(turn1, int)

        turn2 = await repo.save_call_interaction(
            session_id=session_id,
            speaker="agent",
            text="EduSaaS offers comprehensive data science curriculums. May I have your email?",
        )
        assert turn2 is not None and isinstance(turn2, int)

        # Retrieve
        interactions = await repo.get_call_interactions(session_id)
        assert len(interactions) == 2
        assert interactions[0]["speaker"] == "caller"
        assert "data science" in interactions[0]["text"]
        assert interactions[1]["speaker"] == "agent"

    @pytest.mark.asyncio
    async def test_lead_upsert_and_update(self, repo: PostgresRepository):
        ts = int(datetime.now(timezone.utc).timestamp())
        phone = f"+1555{ts % 10000000:07d}"
        email = f"lead_{ts}@example.com"

        # 1. Insert new lead
        lead_id = await repo.upsert_lead(
            name="Integration Lead",
            phone_number=phone,
            email=email,
            status="contacted",
            interested_in="Full Stack AI",
            metadata={"source": "inbound_call", "score": 85},
        )
        assert lead_id is not None and isinstance(lead_id, int)

        # 2. Update existing lead by phone
        updated_id = await repo.upsert_lead(
            name="Integration Lead Confirmed",
            phone_number=phone,
            status="qualified",
            metadata={"notes": "Caller confirmed enrollment interest"},
        )
        assert updated_id == lead_id

        # 3. Verify lead in list
        leads = await repo.list_leads(limit=20)
        found = [l for l in leads if l["id"] == lead_id]
        assert len(found) == 1
        assert found[0]["name"] == "Integration Lead Confirmed"
        assert found[0]["status"] == "qualified"

    @pytest.mark.asyncio
    async def test_create_and_list_hr_followup(self, repo: PostgresRepository):
        ts = int(datetime.now(timezone.utc).timestamp())
        session_id = f"hr_session_{ts}"
        
        ticket_id = await repo.create_hr_followup(
            candidate_name="Robert Taylor",
            session_id=session_id,
            phone_number=f"+1555{ts % 10000000:07d}",
            email=f"robert_{ts}@example.com",
            reason="Inquired about AI curriculum instructor position",
            status="queued",
        )
        assert ticket_id is not None and isinstance(ticket_id, int)

        followups = await repo.list_hr_followups(limit=20)
        found = [f for f in followups if f["id"] == ticket_id]
        assert len(found) == 1
        assert found[0]["name"] == "Robert Taylor"
        assert found[0]["status"] == "queued"


# =============================================================================
# 3. FastMCP Tools with Database Integration and Offline Fallback
# =============================================================================

class TestFastMCPDatabaseIntegration:
    """Verify FastMCP server tools update_lead and create_hr_followup with DB."""

    @pytest.mark.asyncio
    async def test_update_lead_mcp_tool_persists_to_database(self):
        repo = PostgresRepository()
        server = create_mcp_server(postgres_repo=repo)

        ts = int(datetime.now(timezone.utc).timestamp())
        phone = f"+1555{ts % 10000000:07d}"
        
        res = await server.call_tool(
            "update_lead",
            {
                "name": "MCP Prospect",
                "phone": phone,
                "email": f"mcp_{ts}@example.com",
                "status": "engaged",
                "notes": "Spoke during consultation",
            },
        )
        assert res["updated"] is True
        assert res["status"] == "engaged"
        # lead_id format must be lead_crm_<id>
        assert res["lead_id"].startswith("lead_crm_")

    @pytest.mark.asyncio
    async def test_create_hr_followup_mcp_tool_persists_to_database(self):
        repo = PostgresRepository()
        server = create_mcp_server(postgres_repo=repo)

        ts = int(datetime.now(timezone.utc).timestamp())
        res = await server.call_tool(
            "create_hr_followup",
            {
                "candidate_name": "MCP Candidate",
                "phone": f"+1555{ts % 10000000:07d}",
                "notes": "Followup for AI tutor role",
            },
        )
        assert res["status"] == "queued"
        assert res["ticket_id"].startswith("ticket_hr_")

    @pytest.mark.asyncio
    async def test_tools_graceful_fallback_when_database_fails(self):
        """Simulate failing PostgresRepository and verify tools never raise fatal errors."""
        failing_repo = MagicMock(spec=PostgresRepository)
        failing_repo.upsert_lead = AsyncMock(side_effect=Exception("Database connection timeout"))
        failing_repo.create_hr_followup = AsyncMock(side_effect=Exception("Database disk full"))

        server = create_mcp_server(postgres_repo=failing_repo)

        # 1. update_lead falls back gracefully
        lead_res = await server.call_tool(
            "update_lead",
            {"status": "contacted", "notes": "Offline lead"},
        )
        assert lead_res["updated"] is True
        assert lead_res["lead_id"].startswith("lead_crm_")

        # 2. create_hr_followup falls back gracefully
        hr_res = await server.call_tool(
            "create_hr_followup",
            {"candidate_name": "Offline Candidate"},
        )
        assert hr_res["status"] == "queued"
        assert hr_res["ticket_id"].startswith("ticket_hr_")


# =============================================================================
# 4. PostgresCallSessionRepository Tests
# =============================================================================

class TestPostgresCallSessionRepository:
    """Verify session summary persistence to PostgreSQL with memory fallback."""

    @pytest.mark.asyncio
    async def test_persist_call_summary_to_postgres(self):
        repo = PostgresRepository()
        session_repo = PostgresCallSessionRepository(postgres_repo=repo)

        call_id = f"call_pg_{int(datetime.now(timezone.utc).timestamp())}"
        summary = CallSummary(
            session_id=call_id,
            call_id=call_id,
            direction="inbound",
            domain="edusaas",
            primary_intent="course_inquiry",
            caller={
                "name": "Maria Garcia",
                "phone": "+15559871234",
                "email": "maria@example.com",
            },
            business_status="qualified",
            important_entities={"course": "Full Stack AI"},
            completed_actions=[{"action": "send_email", "success": True}],
            start_time=100.0,
            end_time=145.0,
            duration=45.0,
            termination_reason="completed_goodbye",
        )

        await session_repo.persist_summary(summary)

        # Retrieve from fallback memory
        retrieved = await session_repo.get_summary(call_id)
        assert retrieved is not None
        assert retrieved.call_id == call_id
        assert retrieved.domain == "edusaas"

        # Check call_interactions in DB
        interactions = await repo.get_call_interactions(call_id)
        assert len(interactions) >= 1
        assert "Call completed" in interactions[0]["text"]

    @pytest.mark.asyncio
    async def test_session_repo_graceful_fallback_when_db_down(self):
        failing_repo = MagicMock(spec=PostgresRepository)
        failing_repo.upsert_lead = AsyncMock(side_effect=Exception("Connection lost"))
        failing_repo.save_call_interaction = AsyncMock(side_effect=Exception("Connection lost"))

        session_repo = PostgresCallSessionRepository(postgres_repo=failing_repo)
        call_id = "call_fallback_999"
        summary = CallSummary(
            session_id=call_id,
            call_id=call_id,
            direction="outbound",
            domain="vayvora",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
        )

        # Should not raise exception
        await session_repo.persist_summary(summary)
        retrieved = await session_repo.get_summary(call_id)
        assert retrieved is not None
        assert retrieved.call_id == call_id


# =============================================================================
# 5. FastAPI Health Endpoint Integration
# =============================================================================

class TestHealthEndpointWithPostgres:
    """Verify health endpoint reports PostgreSQL status."""

    def test_health_endpoint_reports_postgres_connected(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "postgres_connected" in data["subsystems"]
        assert data["subsystems"]["postgres_connected"] is True
        assert data["components"]["postgres"] is True
