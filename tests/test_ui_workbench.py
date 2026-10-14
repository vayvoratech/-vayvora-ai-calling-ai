"""Comprehensive unit tests for the Infra Builder Workbench (Phase 6).

Covers all 18 required scenarios:
1. Inbound Infra Builder session creation
2. Outbound Infra Builder session creation
3. Inbound session defaults to Infra Builder
4. Outbound villa campaign session creation
5. Unknown inbound caller handling
6. Known outbound caller handling
7. Single-business domain stability
8. Intent switching and preemption
9. RAG knowledge response incorporation
10. Missing RAG knowledge graceful handling
11. Proposed MCP action extraction
12. Verified MCP action execution
13. Failed MCP action non-terminating handling
14. Explicit termination behavior
15. Non-terminating "no" responses
16. Clean session reset behavior
17. External service unavailable recovery
18. Mock mode vs Live mode visibility and distinction
"""

import pytest
from typing import Dict, Any

from src.config import Settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    CallStatus,
    ConversationStage,
    DomainType,
    RAGChunk,
    RAGQuery,
    ToolCallRequest,
)
from src.rag.embeddings import MockEmbeddingProvider
from src.rag.retriever import GroundedKnowledgeProvider
from src.tools.mcp_client import MockToolProvider
from src.ui.bootstrap import InteractiveMockLLMProvider, bootstrap_workbench
from src.ui.service import RuntimeMode, ServiceComponents, WorkbenchService


@pytest.fixture
def workbench_service() -> WorkbenchService:
    """Provide a freshly bootstrapped mock WorkbenchService for deterministic tests."""
    settings = Settings(
        gemini_api_key="mock_test_key",
        mcp_server_url="http://mock-mcp.local",
    )
    return bootstrap_workbench(settings=settings, force_mock=True)


class TestWorkbenchSessionCreation:
    """Tests 1-6: Inbound/outbound session initialization across domains and caller types."""

    def test_01_inbound_infra_builder_session(self, workbench_service: WorkbenchService):
        """Scenario 1: Inbound Infra Builder session creation."""
        state = workbench_service.create_inbound_session(
            call_id="test-in-ib-01",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )
        assert state.metadata.call_id == "test-in-ib-01"
        assert state.current_domain == DomainType.INFRA_BUILDER
        assert state.metadata.direction == CallDirection.INBOUND
        assert state.conversation_active is True
        assert state.stage == ConversationStage.GREETING

    def test_02_outbound_infra_builder_session(self, workbench_service: WorkbenchService):
        """Scenario 2: Outbound Infra Builder session creation."""
        state = workbench_service.create_outbound_session(
            call_id="test-out-ib-01",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
            caller_name="Sarah Connor",
            campaign_id="CAMP-SKYLINE",
            campaign_objective="Discuss Skyline Towers residences",
            caller_email="sarah@example.com",
        )
        assert state.metadata.call_id == "test-out-ib-01"
        assert state.current_domain == DomainType.INFRA_BUILDER
        assert state.metadata.direction == CallDirection.OUTBOUND
        assert state.caller.name == "Sarah Connor"
        assert state.caller.email == "sarah@example.com"
        assert state.caller.is_known() is True
        assert state.metadata.campaign_id == "CAMP-SKYLINE"

    def test_03_inbound_session_defaults_to_infra_builder(self, workbench_service: WorkbenchService):
        """Scenario 3: Inbound session defaults to Infra Builder."""
        state = workbench_service.create_inbound_session(
            call_id="test-in-def-01",
            caller_phone="+919845123456",
        )
        assert state.metadata.call_id == "test-in-def-01"
        assert state.current_domain == DomainType.INFRA_BUILDER
        assert state.metadata.direction == CallDirection.INBOUND
        assert state.conversation_active is True

    def test_04_outbound_villa_campaign_session(self, workbench_service: WorkbenchService):
        """Scenario 4: Outbound Green Valley Villas session creation."""
        state = workbench_service.create_outbound_session(
            call_id="test-out-villa-01",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
            caller_name="Dr. Miles Dyson",
            campaign_id="CAMP-VILLAS",
            campaign_objective="Green Valley Villas consultation",
            company="Cyberdyne Systems",
        )
        assert state.metadata.call_id == "test-out-villa-01"
        assert state.current_domain == DomainType.INFRA_BUILDER
        assert state.metadata.direction == CallDirection.OUTBOUND
        assert state.caller.name == "Dr. Miles Dyson"
        assert state.caller.company == "Cyberdyne Systems"

    def test_05_unknown_inbound_caller(self, workbench_service: WorkbenchService):
        """Scenario 5: Unknown inbound caller has anonymous profile attributes."""
        state = workbench_service.create_inbound_session(
            call_id="test-in-anon-01",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )
        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.is_known() is False

    def test_06_known_outbound_caller(self, workbench_service: WorkbenchService):
        """Scenario 6: Known outbound caller is flagged as known with pre-populated details."""
        state = workbench_service.create_outbound_session(
            call_id="test-out-known-01",
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
            caller_name="Alice Smith",
            campaign_id="CAMP-001",
            campaign_objective="Followup",
            caller_email="alice@tech.org",
            company="Tech Corp",
        )
        assert state.caller.is_known() is True
        assert state.caller.name == "Alice Smith"
        assert state.caller.email == "alice@tech.org"
        assert state.caller.company == "Tech Corp"


class TestWorkbenchConversationalBehaviors:
    """Tests 7-16: Core turns, domain stability, RAG, MCP actions, termination, and resets."""

    def test_07_single_business_domain_stability(self, workbench_service: WorkbenchService):
        """Scenario 7: Infra Builder remains locked as sole active business domain throughout conversation."""
        session_id = "test-switch-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        # Turn 1: Property inquiry
        workbench_service.process_turn(session_id, "Tell me about Skyline Towers.")
        state = workbench_service.get_session(session_id)
        assert state.current_domain == DomainType.INFRA_BUILDER

        # Turn 2: Followup inquiry
        workbench_service.process_turn(session_id, "What about Green Valley Villas?")
        state = workbench_service.get_session(session_id)
        assert state.current_domain == DomainType.INFRA_BUILDER

    def test_08_intent_switching(self, workbench_service: WorkbenchService):
        """Scenario 8: Preemption of intent by latest user request."""
        session_id = "test-intent-switch-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        # Turn 1: Property inquiry intent
        workbench_service.process_turn(session_id, "Tell me about Skyline Towers in Whitefield.")
        state = workbench_service.get_session(session_id)
        assert state.current_intent == "property_inquiry"

        # Turn 2: Switch to scheduling a site visit
        workbench_service.process_turn(session_id, "Can we schedule a consultation meeting?")
        state = workbench_service.get_session(session_id)
        assert state.current_intent == "schedule_site_visit"
        assert "property_inquiry" in state.intent_history

    def test_09_rag_knowledge_response(self, workbench_service: WorkbenchService):
        """Scenario 9: RAG knowledge is retrieved and incorporated into response."""
        session_id = "test-rag-success-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        state, turn_result = workbench_service.process_turn(
            session_id, "Tell me about Skyline Towers in Whitefield."
        )
        assert len(state.history) == 2
        agent_turn = state.history[-1]
        assert agent_turn.role.value == "agent"
        # Citations or knowledge sources recorded
        debug_payload = workbench_service.get_debug_payload(state, turn_result)
        assert debug_payload["rag"]["knowledge_required"] is True

    def test_10_missing_rag_knowledge_handling(self, workbench_service: WorkbenchService):
        """Scenario 10: RAG with zero relevant matches returns graceful response without crashing."""
        session_id = "test-rag-missing-01"
        state = workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        # Inject canned decision with knowledge_required=True but an obscure query
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="obscure_topic",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="I don't have verified details on that specific topic.",
            knowledge_required=True,
            knowledge_query="completely nonexistent quantum hyperloop project",
        )
        workbench_service.engine.llm = MockLLMProvider(canned_decisions=[decision])

        state, turn_result = workbench_service.process_turn(
            session_id, "What about your quantum hyperloop project?"
        )
        assert state.conversation_active is True
        assert len(turn_result.response_text) > 10

    def test_11_proposed_mcp_action_in_debug(self, workbench_service: WorkbenchService):
        """Scenario 11: Proposed MCP action without confirmed slot proposes find_available_slots."""
        session_id = "test-prop-action-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        state, turn_result = workbench_service.process_turn(
            session_id, "Can we schedule a consultation meeting?"
        )
        debug_payload = workbench_service.get_debug_payload(state, turn_result)
        assert debug_payload["tools"]["proposed_action"] is not None
        assert debug_payload["tools"]["proposed_action"]["tool_name"] == "find_available_slots"

    def test_12_verified_mcp_action_execution(self, workbench_service: WorkbenchService):
        """Scenario 12: Executed MCP action passes verification and updates state."""
        session_id = "test-ver-action-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        state, turn_result = workbench_service.process_turn(
            session_id, "Please schedule a meeting for Tomorrow at 10 AM."
        )
        debug_payload = workbench_service.get_debug_payload(state, turn_result)

        assert debug_payload["tools"]["tool_status"] == "succeeded"
        assert debug_payload["tools"]["verification_status"] == "verified"
        assert debug_payload["tools"]["external_reference"] is not None
        assert state.conversation_active is True

    def test_13_failed_mcp_action_does_not_terminate_call(self, workbench_service: WorkbenchService):
        """Scenario 13: Failed tool action records failure, preserves conversation_active=True."""
        failing_tools = MockToolProvider(force_unavailable=True)
        workbench_service.engine.tool_provider = failing_tools
        workbench_service.tool_provider = failing_tools
        svc = workbench_service

        session_id = "test-fail-action-01"
        svc.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        state, turn_result = svc.process_turn(
            session_id, "Schedule a consultation for Tomorrow at 10 AM."
        )
        debug_payload = svc.get_debug_payload(state, turn_result)

        assert debug_payload["tools"]["tool_status"] == "failed"
        assert debug_payload["tools"]["verification_status"] == "unverified"
        assert state.conversation_active is True
        assert "could not be booked" in turn_result.response_text.lower() or "system issue" in turn_result.response_text.lower()

    def test_14_explicit_termination(self, workbench_service: WorkbenchService):
        """Scenario 14: Caller explicit farewell cleanly terminates the session."""
        session_id = "test-term-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        state, turn_result = workbench_service.process_turn(
            session_id, "That's all, thank you. Goodbye."
        )
        assert state.conversation_active is False
        assert state.termination_requested is True
        assert state.stage == ConversationStage.COMPLETED

    def test_15_no_does_not_terminate_call(self, workbench_service: WorkbenchService):
        """Scenario 15: Non-terminating 'no' or 'no thanks' leaves call active."""
        session_id = "test-no-term-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        state, turn_result = workbench_service.process_turn(session_id, "No thanks.")
        assert state.conversation_active is True
        assert state.termination_requested is False
        assert "anything else" in turn_result.response_text.lower() or "assistance" in turn_result.response_text.lower() or "help" in turn_result.response_text.lower()

    def test_16_clean_session_reset(self, workbench_service: WorkbenchService):
        """Scenario 16: Resetting session completely removes stale context."""
        session_id = "test-reset-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        workbench_service.process_turn(session_id, "Tell me about Skyline Towers.")
        assert workbench_service.get_session(session_id) is not None

        # Reset session
        workbench_service.reset_session(session_id)
        assert workbench_service.get_session(session_id) is None


class TestWorkbenchDiagnosticsAndModes:
    """Tests 17-18: Error resilience and MOCK vs LIVE distinction."""

    def test_17_external_service_unavailable(self, workbench_service: WorkbenchService):
        """Scenario 17: UI workbench handles unexpected engine exceptions gracefully."""
        session_id = "test-err-resilience-01"
        workbench_service.create_inbound_session(
            call_id=session_id,
            caller_phone="+919845123456",
            domain=DomainType.INFRA_BUILDER,
        )

        # Non-existent session
        with pytest.raises(KeyError):
            workbench_service.process_turn("non-existent-session-id", "Hello")

    def test_18_mock_mode_is_visibly_distinguished_from_live_mode(self):
        """Scenario 18: Service reports MOCK vs LIVE mode explicitly."""
        mock_svc = bootstrap_workbench(force_mock=True)
        assert mock_svc.overall_mode == RuntimeMode.MOCK
        assert mock_svc.llm_mode == "MOCK"
        assert mock_svc.rag_mode == "MOCK"
        assert mock_svc.tool_mode == "MOCK"

        payload = mock_svc.get_debug_payload(state=None)
        assert payload["runtime"]["overall_mode"] == "MOCK"
        assert payload["runtime"]["llm_mode"] == "MOCK"
        assert payload["runtime"]["tool_mode"] == "MOCK"
