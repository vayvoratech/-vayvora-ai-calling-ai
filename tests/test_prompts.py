"""Unit tests for Dynamic Prompt Synthesizer."""

import pytest
from src.core.prompts import PromptSynthesizer
from src.core.types import CallDirection, DomainType, TurnRole
from src.domains import EDUSAAS_DOMAIN_CONFIG, VAYVORA_DOMAIN_CONFIG
from src.state.manager import ConversationStateManager


class TestPromptSynthesizer:
    """Verify dynamic prompt composition across domains, directions, and states."""

    def test_stable_rules_present_in_system_prompt(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-p1", "+15551234567", domain=DomainType.EDUSAAS)

        system_prompt = synthesizer.build_system_instruction(state, EDUSAAS_DOMAIN_CONFIG)
        assert "INTENT PRIMACY & PREEMPTION" in system_prompt
        assert "STRICT GROUNDING & UNTRUSTED DATA" in system_prompt
        assert "EXTERNAL ACTIONS & VERIFICATION" in system_prompt
        assert "CONVERSATION CONTINUITY" in system_prompt
        assert "OUTPUT FORMAT" in system_prompt

    def test_edusaas_domain_metadata_injected(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-p2", "+15551234567", domain=DomainType.EDUSAAS)

        system_prompt = synthesizer.build_system_instruction(state, EDUSAAS_DOMAIN_CONFIG)
        assert "EduSaaS" in system_prompt
        assert "course_information" in system_prompt
        assert "student_name" in system_prompt

    def test_vayvora_domain_metadata_injected(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-p3", "+15551234567", domain=DomainType.VAYVORA)

        system_prompt = synthesizer.build_system_instruction(state, VAYVORA_DOMAIN_CONFIG)
        assert "Vayvora" in system_prompt
        assert "company_location" in system_prompt
        assert "technical_requirement" in system_prompt

    def test_outbound_known_caller_details_with_do_not_ask_rule(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-out-p4",
            caller_phone="+15559990000",
            domain=DomainType.EDUSAAS,
            caller_name="Rohan Gupta",
            campaign_id="edusaas-ai-diploma",
            campaign_objective="Discuss AI Engineering enrollment and scholarships",
            caller_email="rohan.gupta@example.com",
            company="XYZ Institute",
        )

        system_prompt = synthesizer.build_system_instruction(state, EDUSAAS_DOMAIN_CONFIG)
        assert "OUTBOUND OUTREACH CALL" in system_prompt
        assert "Rohan Gupta" in system_prompt
        assert "rohan.gupta@example.com" in system_prompt
        assert "DO NOT ask the caller to provide their name, email, phone, or company" in system_prompt

    def test_inbound_unknown_caller_details(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-in-p5", "+15551234567", domain=DomainType.EDUSAAS)

        system_prompt = synthesizer.build_system_instruction(state, EDUSAAS_DOMAIN_CONFIG)
        assert "INBOUND CALL" in system_prompt
        assert "Anonymous / Not yet identified" in system_prompt
        assert "+15551234567" in system_prompt

    def test_user_prompt_includes_dialogue_history(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-hist-p6", "+15551234567", domain=DomainType.EDUSAAS)
        state.record_turn(TurnRole.CALLER, "What courses do you offer?")
        state.record_turn(TurnRole.AGENT, "We offer AI, Full-Stack, and Data Science programs.")

        user_prompt = synthesizer.build_user_prompt(state, "How long is the AI course?")
        assert "What courses do you offer?" in user_prompt
        assert "We offer AI, Full-Stack, and Data Science programs." in user_prompt
        assert "How long is the AI course?" in user_prompt
