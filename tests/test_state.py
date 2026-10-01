"""Comprehensive deterministic tests for conversation state and lifecycle management.

Tests all 22 core state behaviors specified in Phase 2 requirements.
"""

import pytest
from pydantic import ValidationError

from src.core.types import (
    CallDirection,
    CallMetadata,
    CallStatus,
    ConversationStage,
    DomainType,
    ToolExecutionResult,
    TurnRole,
)
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState


class TestConversationState:
    """Test suite covering the 22 deterministic conversation state specifications."""

    # 1. New conversation state
    def test_01_new_conversation_state(self):
        metadata = CallMetadata(
            call_id="call-001",
            direction=CallDirection.INBOUND,
            primary_domain=DomainType.EDUSAAS,
            caller_phone="+15551234567",
        )
        state = ConversationState(
            metadata=metadata,
            current_domain=DomainType.EDUSAAS,
            primary_domain=DomainType.EDUSAAS,
        )
        assert state.metadata.call_id == "call-001"
        assert state.stage == ConversationStage.GREETING
        assert state.conversation_active is True
        assert state.business_status == "new"
        assert state.current_intent is None
        assert state.intent_history == []
        assert state.history == []

    # 2. Inbound state
    def test_02_inbound_state_creation(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="inbound-101",
            caller_phone="+15552345678",
            domain=DomainType.EDUSAAS,
        )
        assert state.metadata.direction == CallDirection.INBOUND
        assert state.current_domain == DomainType.EDUSAAS
        assert state.primary_domain == DomainType.EDUSAAS
        assert state.conversation_active is True
        assert mgr.get("inbound-101") is state

    # 3. Outbound state
    def test_03_outbound_state_creation(self):
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="outbound-201",
            caller_phone="+15559876543",
            domain=DomainType.VAYVORA,
            caller_name="Vikram Mehta",
            campaign_id="vayvora-q3-ai-demo",
            campaign_objective="Schedule technical discovery demo",
            caller_email="vikram@enterprise.com",
            company="Enterprise AI Solutions",
            known_purpose="Enterprise Voice AI requirements",
        )
        assert state.metadata.direction == CallDirection.OUTBOUND
        assert state.metadata.campaign_id == "vayvora-q3-ai-demo"
        assert state.caller.name == "Vikram Mehta"
        assert state.caller.company == "Enterprise AI Solutions"
        assert state.caller.campaign_objective == "Schedule technical discovery demo"
        assert state.business_status == "outreach_initiated"
        assert state.conversation_active is True

    # 4. Known caller information
    def test_04_known_caller_information(self):
        caller = CallerProfile(
            name="Ananya Roy",
            phone="+15553334444",
            email="ananya@example.com",
            company="Tech Corp",
        )
        assert caller.is_known() is True
        assert caller.has_contact_info() is True
        assert caller.name == "Ananya Roy"
        assert caller.email == "ananya@example.com"

    # 5. Unknown caller information
    def test_05_unknown_caller_information(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="inbound-anon",
            caller_phone="+15550001111",
        )
        assert state.caller.name is None
        assert state.caller.email is None
        assert state.caller.is_known() is False
        # Agent can incrementally update them later
        state.update_slot("student_name", "Rahul Sen")
        state.update_slot("email", "rahul@example.com")
        assert state.caller.name == "Rahul Sen"
        assert state.caller.email == "rahul@example.com"
        assert state.caller.is_known() is True

    # 6. EduSaaS domain state
    def test_06_edusaas_domain_state(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="edu-001",
            caller_phone="+15554445555",
            domain=DomainType.EDUSAAS,
        )
        assert state.current_domain == DomainType.EDUSAAS
        state.set_intent("course_information")
        assert state.current_intent == "course_information"

    # 7. Vayvora domain state
    def test_07_vayvora_domain_state(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state(
            call_id="vay-001",
            caller_phone="+15556667777",
            domain=DomainType.VAYVORA,
        )
        assert state.current_domain == DomainType.VAYVORA
        state.set_intent("career_information")
        assert state.current_intent == "career_information"

    # 8. Intent update
    def test_08_intent_update(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-int-1", "+15551112222")
        state.set_intent("course_recommendation")
        assert state.current_intent == "course_recommendation"
        assert state.intent_history == []

    # 9. Intent history preservation
    def test_09_intent_history_preservation(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-hist-1", "+15551112222")
        state.set_intent("course_information")
        state.set_intent("pricing")
        state.set_intent("duration")

        assert state.current_intent == "duration"
        assert state.intent_history == ["course_information", "pricing"]

    # 10. Intent preemption
    def test_10_intent_preemption(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-preempt-1", "+15551112222")
        # Agent asks a question
        state.set_pending_question("Which programming languages do you know?")
        state.set_intent("course_recommendation")

        # User suddenly asks a company question instead of answering
        state.set_intent("company_location", clear_pending_question=True)

        assert state.current_intent == "company_location"
        assert state.pending_question is None
        assert "course_recommendation" in state.intent_history

    # 11. Domain switching
    def test_11_domain_switching(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-switch-1", "+15551112222", domain=DomainType.EDUSAAS)
        state.set_intent("course_information")
        state.update_slot("interest", "AI Engineering")

        # Caller asks about Vayvora corporate services
        state.switch_domain(DomainType.VAYVORA)
        state.set_intent("product_information")

        assert state.primary_domain == DomainType.EDUSAAS  # Preserved original entry domain
        assert state.current_domain == DomainType.VAYVORA   # Switched active domain
        assert state.current_intent == "product_information"
        assert state.intent_history == ["course_information"]
        assert state.extracted_slots["interest"] == "AI Engineering"

    # 12. Conversation stage updates
    def test_12_conversation_stage_updates(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-stage-1", "+15551112222")
        assert state.stage == ConversationStage.GREETING

        state.set_stage(ConversationStage.DISCOVERY)
        assert state.stage == ConversationStage.DISCOVERY

        state.set_stage(ConversationStage.INFORMATION)
        assert state.stage == ConversationStage.INFORMATION

    # 13. Slot/entity updates
    def test_13_slot_entity_updates(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-slots-1", "+15551112222")

        state.update_slot("student_name", "Kavita Rao")
        state.update_slot("education", "B.Sc Computer Science")
        state.update_slot("year", "2024")
        state.update_slot("target_course", "Applied AI Bootcamp")

        assert state.extracted_slots["student_name"] == "Kavita Rao"
        assert state.extracted_slots["education"] == "B.Sc Computer Science"
        assert state.extracted_slots["year"] == "2024"
        assert state.extracted_slots["target_course"] == "Applied AI Bootcamp"
        assert state.caller.name == "Kavita Rao"
        assert state.current_interest == "Applied AI Bootcamp"

    # 14. Pending question
    def test_14_pending_question(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-pq-1", "+15551112222")
        assert state.pending_question is None

        state.set_pending_question("Are you looking for weekend or weekday batches?")
        assert state.pending_question == "Are you looking for weekend or weekday batches?"

        # Answering clears it
        state.set_pending_question(None)
        assert state.pending_question is None

    # 15. Pending action
    def test_15_pending_action(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-pa-1", "+15551112222")
        assert state.pending_action is None

        state.set_pending_action("tools_email_send")
        assert state.pending_action == "tools_email_send"

        tool_result = ToolExecutionResult(
            tool_name="tools_email_send",
            success=True,
            data={"status": "sent"},
            verification_code="EMAIL-OK-2026",
        )
        state.complete_action("tools_email_send", tool_result)

        assert state.pending_action is None
        assert state.last_action == "tools_email_send"
        assert state.last_tool_result == tool_result
        assert state.conversation_active is True  # Remains active!

    # 16. Business status independent from conversation status
    def test_16_business_status_independent(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-bs-1", "+15551112222")
        assert state.business_status == "new"
        assert state.conversation_active is True

        state.update_business_status("qualified")
        assert state.business_status == "qualified"
        assert state.conversation_active is True

        state.update_business_status("not_interested")
        assert state.business_status == "not_interested"
        assert state.conversation_active is True  # Still open unless caller hangs up

    # 17. Interested lead remains active
    def test_17_interested_lead_remains_active(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-lead-1", "+15551112222")

        state.update_business_status("interested")
        assert state.business_status == "interested"
        assert state.conversation_active is True  # Call must NOT terminate

    # 18. Explicit termination state
    def test_18_explicit_termination_state(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-term-1", "+15551112222")
        assert state.conversation_active is True
        assert state.termination_requested is False

        state.request_termination(reason="caller_said_goodbye")

        assert state.termination_requested is True
        assert state.conversation_active is False
        assert state.termination_reason == "caller_said_goodbye"
        assert state.stage == ConversationStage.COMPLETED
        assert state.metadata.status == CallStatus.COMPLETED
        assert state.metadata.end_time is not None

    # 19. "No" does not automatically terminate
    def test_19_no_does_not_automatically_terminate(self):
        assert ConversationState.is_explicit_termination("no") is False
        assert ConversationState.is_explicit_termination("No.") is False
        assert ConversationState.is_explicit_termination("no thanks") is False
        assert ConversationState.is_explicit_termination("not really") is False
        assert ConversationState.is_explicit_termination("no I don't have python experience") is False

        # In contrast, explicit farewell phrases DO terminate
        assert ConversationState.is_explicit_termination("goodbye") is True
        assert ConversationState.is_explicit_termination("Thank you, that's all for today.") is True
        assert ConversationState.is_explicit_termination("I am done, thank you bye") is True

    # 20. State serialization and deserialization
    def test_20_state_serialization_deserialization(self):
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-ser-1",
            caller_phone="+15559998888",
            domain=DomainType.EDUSAAS,
            caller_name="Rohan Verma",
            campaign_id="batch-2026-outreach",
            campaign_objective="Course enrollment confirmation",
        )
        state.set_intent("enrollment")
        state.update_slot("target_course", "Full Stack AI")
        state.record_turn(TurnRole.AGENT, "Hello Rohan, calling regarding your enrollment.")
        state.record_turn(TurnRole.CALLER, "Yes, I would like to confirm the schedule.")

        dumped = state.model_dump_json()
        restored = ConversationState.model_validate_json(dumped)

        assert restored.metadata.call_id == state.metadata.call_id
        assert restored.caller.name == "Rohan Verma"
        assert restored.current_intent == "enrollment"
        assert restored.extracted_slots["target_course"] == "Full Stack AI"
        assert len(restored.history) == 2
        assert restored.history[0].content == "Hello Rohan, calling regarding your enrollment."

    # 21. State copy/update behavior (cloning)
    def test_21_state_copy_behavior(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-clone-1", "+15551234567")
        state.update_slot("experience", "2 years")

        snapshot = state.clone()
        assert snapshot.extracted_slots["experience"] == "2 years"

        # Mutate original
        state.update_slot("experience", "5 years")
        state.set_intent("course_comparison")

        # Snapshot remains unchanged
        assert snapshot.extracted_slots["experience"] == "2 years"
        assert snapshot.current_intent is None

    # 22. Previous context remains available after intent switch
    def test_22_previous_context_remains_available(self):
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-ctx-1", "+15551112222", domain=DomainType.EDUSAAS)

        # 1. Talk about course
        state.set_intent("course_information")
        state.update_slot("target_course", "Generative AI Systems")
        state.record_turn(TurnRole.CALLER, "Tell me about the Generative AI course.")
        state.record_turn(TurnRole.AGENT, "It covers transformers, LLMs, and agentic workflows.")

        # 2. Topic shift to Vayvora jobs
        state.switch_domain(DomainType.VAYVORA)
        state.set_intent("career_information")
        state.record_turn(TurnRole.CALLER, "Wait, does Vayvora hire graduates for AI engineer roles?")
        state.record_turn(TurnRole.AGENT, "Yes, Vayvora actively recruits AI and LLM engineers.")

        # 3. Caller returns to course
        state.switch_domain(DomainType.EDUSAAS)
        state.set_intent("course_information")

        # Verify historical context is intact
        assert state.intent_history == ["course_information", "career_information"]
        assert state.extracted_slots["target_course"] == "Generative AI Systems"
        assert len(state.history) == 4
        assert state.history[0].content == "Tell me about the Generative AI course."
        assert state.history[2].content == "Wait, does Vayvora hire graduates for AI engineer roles?"
