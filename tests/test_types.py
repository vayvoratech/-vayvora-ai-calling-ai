"""Unit tests for foundational Pydantic types and data contracts."""

import pytest
from pydantic import ValidationError

from src.core.types import (
    CallDirection,
    CallMetadata,
    CallStatus,
    DialogueTurn,
    DomainType,
    RAGChunk,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
    TurnRole,
)


class TestEnums:
    """Test that all core enumerations define expected domains and states."""

    def test_domain_types(self):
        assert DomainType.EDUSAAS == "edusaas"
        assert DomainType.VAYVORA == "vayvora"
        assert DomainType.GENERAL == "general"

    def test_call_directions(self):
        assert CallDirection.INBOUND == "inbound"
        assert CallDirection.OUTBOUND == "outbound"

    def test_call_statuses(self):
        assert CallStatus.RINGING == "ringing"
        assert CallStatus.ACTIVE == "active"
        assert CallStatus.ON_HOLD == "on_hold"
        assert CallStatus.COMPLETED == "completed"
        assert CallStatus.FAILED == "failed"

    def test_turn_roles(self):
        assert TurnRole.CALLER == "caller"
        assert TurnRole.AGENT == "agent"
        assert TurnRole.SYSTEM == "system"
        assert TurnRole.TOOL == "tool"


class TestDialogueTurn:
    """Test DialogueTurn model creation, validation, and serialization."""

    def test_valid_turn_creation(self):
        turn = DialogueTurn(
            role=TurnRole.CALLER,
            content="Hello, I want to learn about AI courses.",
        )
        assert turn.role == TurnRole.CALLER
        assert turn.content == "Hello, I want to learn about AI courses."
        assert turn.timestamp > 0
        assert turn.grounded_citations == []
        assert turn.tool_calls_executed == []
        assert turn.intent_detected is None

    def test_turn_with_metadata(self):
        turn = DialogueTurn(
            role=TurnRole.AGENT,
            content="We offer a comprehensive AI Engineering diploma.",
            grounded_citations=["edusaas_ai_curriculum_v1"],
            tool_calls_executed=["lookup_course_schedule"],
            intent_detected="inquire_course_details",
        )
        assert len(turn.grounded_citations) == 1
        assert "lookup_course_schedule" in turn.tool_calls_executed
        assert turn.intent_detected == "inquire_course_details"

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            DialogueTurn(role=TurnRole.CALLER, content="")

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            DialogueTurn(role="unauthorized_role", content="Hello")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            DialogueTurn(
                role=TurnRole.CALLER,
                content="Hello",
                extra_unknown_field="prohibited",
            )

    def test_roundtrip_serialization(self):
        turn = DialogueTurn(
            role=TurnRole.AGENT,
            content="Your consultation is scheduled.",
            grounded_citations=["chunk_123"],
        )
        dumped_json = turn.model_dump_json()
        restored = DialogueTurn.model_validate_json(dumped_json)
        assert restored == turn


class TestCallMetadata:
    """Test CallMetadata model validation and lifecycle."""

    def test_valid_metadata_creation(self):
        meta = CallMetadata(
            call_id="call-uuid-001",
            direction=CallDirection.INBOUND,
            primary_domain=DomainType.EDUSAAS,
            caller_phone="+15551234567",
        )
        assert meta.call_id == "call-uuid-001"
        assert meta.status == CallStatus.RINGING
        assert meta.start_time > 0
        assert meta.end_time is None
        assert meta.campaign_id is None

    def test_outbound_metadata(self):
        meta = CallMetadata(
            call_id="outbound-uuid-002",
            direction=CallDirection.OUTBOUND,
            primary_domain=DomainType.VAYVORA,
            caller_phone="+15559876543",
            caller_name="Alice Smith",
            campaign_id="q3-corporate-outreach",
            outbound_objective="Product introduction and demo discovery",
            status=CallStatus.ACTIVE,
        )
        assert meta.direction == CallDirection.OUTBOUND
        assert meta.caller_name == "Alice Smith"
        assert meta.campaign_id == "q3-corporate-outreach"

    def test_empty_call_id_rejected(self):
        with pytest.raises(ValidationError):
            CallMetadata(
                call_id="",
                direction=CallDirection.INBOUND,
                primary_domain=DomainType.EDUSAAS,
                caller_phone="+15551234567",
            )


class TestRAGModels:
    """Test RAGQuery and RAGChunk models."""

    def test_valid_rag_query(self):
        query = RAGQuery(
            domain=DomainType.EDUSAAS,
            query_text="What are the prerequisites for the Data Science bootcamp?",
            top_k=5,
            relevance_threshold=0.75,
        )
        assert query.top_k == 5
        assert query.relevance_threshold == 0.75

    def test_invalid_rag_query_threshold(self):
        with pytest.raises(ValidationError):
            RAGQuery(
                domain=DomainType.EDUSAAS,
                query_text="Valid query",
                relevance_threshold=1.5,  # Must be <= 1.0
            )

        with pytest.raises(ValidationError):
            RAGQuery(
                domain=DomainType.EDUSAAS,
                query_text="Valid query",
                relevance_threshold=-0.1,  # Must be >= 0.0
            )

    def test_invalid_rag_query_top_k(self):
        with pytest.raises(ValidationError):
            RAGQuery(
                domain=DomainType.EDUSAAS,
                query_text="Valid query",
                top_k=0,  # Must be >= 1
            )

    def test_empty_query_text_rejected(self):
        with pytest.raises(ValidationError):
            RAGQuery(
                domain=DomainType.EDUSAAS,
                query_text="",
            )

    def test_valid_rag_chunk(self):
        chunk = RAGChunk(
            doc_id="chunk-42",
            domain=DomainType.VAYVORA,
            title="Software Engineering Team Overview",
            content="Our AI engineering team develops production agent pipelines.",
            score=0.88,
            metadata={"source_file": "vayvora_overview.md", "version": 2},
        )
        assert chunk.score == 0.88
        assert chunk.metadata["version"] == 2

    def test_invalid_rag_chunk_score(self):
        with pytest.raises(ValidationError):
            RAGChunk(
                doc_id="chunk-42",
                domain=DomainType.VAYVORA,
                title="Title",
                content="Content",
                score=1.1,  # Must be <= 1.0
            )


class TestToolModels:
    """Test ToolCallRequest and ToolExecutionResult."""

    def test_valid_tool_request(self):
        req = ToolCallRequest(
            tool_name="send_course_brochure",
            arguments={"email": "student@example.com", "course_id": "ai-101"},
            call_id="call-uuid-001",
        )
        assert req.tool_name == "send_course_brochure"
        assert req.arguments["email"] == "student@example.com"

    def test_empty_tool_name_rejected(self):
        with pytest.raises(ValidationError):
            ToolCallRequest(
                tool_name="",
                arguments={},
                call_id="call-uuid-001",
            )

    def test_successful_tool_result(self):
        result = ToolExecutionResult(
            tool_name="send_course_brochure",
            success=True,
            data={"message_id": "msg_98234"},
            verification_code="VERIFIED-2026-OK",
        )
        assert result.success is True
        assert result.error_message is None
        assert result.verification_code == "VERIFIED-2026-OK"

    def test_failed_tool_result(self):
        result = ToolExecutionResult(
            tool_name="send_course_brochure",
            success=False,
            error_message="Invalid recipient address",
        )
        assert result.success is False
        assert result.error_message == "Invalid recipient address"
        assert result.verification_code is None
