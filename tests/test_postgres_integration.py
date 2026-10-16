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
from src.database.repository import PostgresRepository, mask_phone_number, normalize_lead_phone
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

    @pytest.mark.asyncio
    async def test_missing_phone_never_generates_synthetic_lead(self, repo: PostgresRepository):
        """Missing or placeholder phone numbers must return None and NEVER create a synthetic lead."""
        leads_before = await repo.list_leads(limit=100)
        count_before = len(leads_before)

        # 1. None phone
        res_none = await repo.upsert_lead(name="No Phone Caller", phone_number=None)
        assert res_none is None

        # 2. Empty string
        res_empty = await repo.upsert_lead(name="Empty Phone Caller", phone_number="")
        assert res_empty is None

        # 3. Whitespace string
        res_ws = await repo.upsert_lead(name="Whitespace Phone", phone_number="   ")
        assert res_ws is None

        # 4. Prohibited placeholder strings
        for ph in ["default caller", "john doe", "unknown", "n/a", "none"]:
            res_ph = await repo.upsert_lead(name="Placeholder", phone_number=ph)
            assert res_ph is None

        # Verify no rows were added or fabricated with +1555...
        leads_after = await repo.list_leads(limit=100)
        assert len(leads_after) == count_before

    @pytest.mark.asyncio
    async def test_get_lead_by_phone_present_and_absent(self, repo: PostgresRepository):
        """Lookup returns correct lead when present and None when absent or invalid."""
        ts = int(datetime.now(timezone.utc).timestamp())
        phone = f"+1555{((ts + 77777) % 9000000 + 1000000):07d}"
        email = f"lead_lookup_{ts}@example.com"

        # 1. Absent lookup returns None
        absent = await repo.get_lead_by_phone(phone)
        assert absent is None

        # 2. Insert lead
        lead_id = await repo.upsert_lead(
            name="Lookup Target",
            phone_number=phone,
            email=email,
            status="contacted",
            interested_in="Pristine City",
            metadata={"priority": "high"},
        )
        assert lead_id is not None

        # 3. Present lookup returns complete record
        found = await repo.get_lead_by_phone(phone)
        assert found is not None
        assert found["id"] == lead_id
        assert found["name"] == "Lookup Target"
        assert found["phone_number"] == phone
        assert found["email"] == email
        assert found["status"] == "contacted"
        assert found["interested_in"] == "Pristine City"
        assert isinstance(found["metadata"], dict)
        assert found["metadata"]["priority"] == "high"

        # 4. Invalid or placeholder lookups return None safely
        assert await repo.get_lead_by_phone(None) is None
        assert await repo.get_lead_by_phone("") is None
        assert await repo.get_lead_by_phone("default caller") is None
        assert await repo.get_lead_by_phone("0000000000") is None

    @pytest.mark.asyncio
    async def test_missing_values_do_not_erase_existing_non_null_details(self, repo: PostgresRepository):
        """Subsequent upserts omitting non-null fields preserve existing values and merge metadata."""
        ts = int(datetime.now(timezone.utc).timestamp())
        phone = f"+1555{ts % 10000000:07d}"

        # 1. Initial insert with full details
        initial_id = await repo.upsert_lead(
            name="Original Name",
            phone_number=phone,
            email="original@example.com",
            status="initial",
            interested_in="Original Project",
            metadata={"field1": "val1", "shared": "old"},
        )
        assert initial_id is not None

        # 2. Update providing only status and new metadata (name, email, interested_in are None)
        updated_id = await repo.upsert_lead(
            name=None,
            phone_number=phone,
            email=None,
            status="updated_status",
            interested_in=None,
            metadata={"field2": "val2", "shared": "new"},
        )
        assert updated_id == initial_id

        # 3. Fetch and verify non-null fields were preserved
        lead = await repo.get_lead_by_phone(phone)
        assert lead is not None
        assert lead["name"] == "Original Name"
        assert lead["email"] == "original@example.com"
        assert lead["interested_in"] == "Original Project"
        assert lead["status"] == "updated_status"
        # Metadata must be merged
        assert lead["metadata"]["field1"] == "val1"
        assert lead["metadata"]["field2"] == "val2"
        assert lead["metadata"]["shared"] == "new"

    @pytest.mark.asyncio
    async def test_invalid_phone_numbers_do_not_trigger_unsafe_identity_matching(self, repo: PostgresRepository):
        """Invalid, dummy repeating, or sequential numbers are rejected from upsert and lookup."""
        invalid_numbers = [
            "0000000000",
            "9999999999",
            "1234567890",
            "0987654321",
            "123",
            "abc-def-ghij",
            "9999",
        ]
        for num in invalid_numbers:
            assert await repo.upsert_lead(name="Bad Phone Lead", phone_number=num) is None
            assert await repo.get_lead_by_phone(num) is None

    def test_phone_normalization_and_masking_utility(self):
        """Phone normalization preserves leading +, strips delimiters, and masks logs."""
        # Normalization
        assert normalize_lead_phone("+1 555-123-4567") == "+15551234567"
        assert normalize_lead_phone("9849012345") == "9849012345"
        assert normalize_lead_phone("+91-98490-12345") == "+919849012345"
        assert normalize_lead_phone("  +15559871234  ") == "+15559871234"
        assert normalize_lead_phone(None) is None
        assert normalize_lead_phone("") is None
        assert normalize_lead_phone("default caller") is None
        assert normalize_lead_phone("0000000000") is None
        assert normalize_lead_phone("1234567890") is None

        # Masking
        assert mask_phone_number("+15551234567") == "***-4567"
        assert mask_phone_number("9849012345") == "***-2345"
        assert mask_phone_number(None) == "<none>"
        assert mask_phone_number("") == "<none>"

    @pytest.mark.asyncio
    async def test_get_latest_interaction_summary(self, repo: PostgresRepository):
        """Retrieve latest summary interaction for a session."""
        session_id = f"test_sum_{int(datetime.now(timezone.utc).timestamp())}"
        await repo.save_call_interaction(session_id, "caller", "Hello, I want details.")
        await repo.save_call_interaction(session_id, "agent", "Here are the details.")
        await repo.save_call_interaction(
            session_id,
            "system",
            "Call completed. Domain: infra_builder, Intent: property_inquiry, Duration: 30s",
        )

        summary = await repo.get_latest_interaction_summary(session_id)
        assert summary is not None
        assert "Call completed." in summary
        assert "infra_builder" in summary

        # Non-existent session returns None
        assert await repo.get_latest_interaction_summary("non_existent_session_id") is None

    def test_sql_uses_parameterized_placeholders(self, repo: PostgresRepository):
        """Verify repository SQL queries use asyncpg $ parameter placeholders."""
        import inspect

        source = inspect.getsource(repo.upsert_lead)
        assert "$1" in source
        assert "$2" in source
        assert "ON CONFLICT (phone_number) DO UPDATE" in source
        # Must not contain format string interpolation into SQL
        assert "f\"INSERT" not in source
        assert "f'''INSERT" not in source
        assert "f\"\"\"INSERT" not in source

        get_source = inspect.getsource(repo.get_lead_by_phone)
        assert "$1" in get_source
        assert "f\"SELECT" not in get_source


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
