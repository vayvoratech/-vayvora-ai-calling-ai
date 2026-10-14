"""Unit tests for Dynamic Prompt Synthesizer."""

import pytest
from src.core.prompts import PromptSynthesizer
from src.core.types import CallDirection, DomainType, TurnRole
from src.domains import INFRA_BUILDER_DOMAIN_CONFIG
from src.state.manager import ConversationStateManager


class TestPromptSynthesizer:
    """Verify dynamic prompt composition across domains, directions, and states."""

    def test_stable_rules_present_in_system_prompt(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-p1", "+15551234567", domain=DomainType.INFRA_BUILDER)

        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
        assert "INTENT PRIMACY & PREEMPTION" in system_prompt
        assert "STRICT GROUNDING & UNTRUSTED DATA" in system_prompt
        assert "EXTERNAL ACTIONS & VERIFICATION" in system_prompt
        assert "CONVERSATION CONTINUITY" in system_prompt
        assert "OUTPUT FORMAT" in system_prompt

    def test_infra_builder_domain_metadata_injected(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-p2", "+15551234567", domain=DomainType.INFRA_BUILDER)

        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
        assert "Infra Builder" in system_prompt
        assert "property_search" in system_prompt
        assert "preferred_project" in system_prompt

    def test_outbound_known_caller_details_with_do_not_ask_rule(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_outbound_state(
            call_id="call-out-p4",
            caller_phone="+15559990000",
            domain=DomainType.INFRA_BUILDER,
            caller_name="Rohan Gupta",
            campaign_id="infra-skyline-launch",
            campaign_objective="Introduce Skyline Towers luxury 3 BHK residences in Whitefield",
            caller_email="rohan.gupta@example.com",
            company="Self-Employed",
        )

        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
        assert "OUTBOUND CALLING — INFRA BUILDER" in system_prompt
        assert "Rohan Gupta" in system_prompt
        assert "Do not ask again for information already available" in system_prompt

    def test_inbound_unknown_caller_details(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-in-p5", "+15551234567", domain=DomainType.INFRA_BUILDER)

        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
        assert "=== CALL DIRECTION: INBOUND ===" in system_prompt
        assert "Anonymous / Not yet identified" in system_prompt
        assert "+15551234567" in system_prompt

    def test_user_prompt_includes_dialogue_history(self):
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-hist-p6", "+15551234567", domain=DomainType.INFRA_BUILDER)
        state.record_turn(TurnRole.CALLER, "What projects do you have in Whitefield?")
        state.record_turn(TurnRole.AGENT, "We have Skyline Towers in Whitefield offering luxury 2 and 3 BHK apartments.")

        user_prompt = synthesizer.build_user_prompt(state, "What is the starting price for a 3 BHK?")
        assert "What projects do you have in Whitefield?" in user_prompt
        assert "We have Skyline Towers in Whitefield offering luxury 2 and 3 BHK apartments." in user_prompt
        assert "What is the starting price for a 3 BHK?" in user_prompt
