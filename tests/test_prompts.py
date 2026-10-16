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
        state = mgr.create_inbound_state("call-in-p5", "+15550009999", domain=DomainType.INFRA_BUILDER)

        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
        assert "=== CALL DIRECTION: INBOUND ===" in system_prompt
        assert "Anonymous / Not yet identified" in system_prompt
        assert "+15550009999" in system_prompt

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

    def test_user_prompt_excludes_redundant_decision_instructions(self):
        """Verify static decision instructions are no longer repeated in per-turn user prompts."""
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-dedup-1", "+15551234567", domain=DomainType.INFRA_BUILDER)

        user_prompt = synthesizer.build_user_prompt(state, "How much does a plot cost?")

        # Assert static instruction headings from decision_instructions.txt are absent
        assert "=== DECISION INSTRUCTIONS ===" not in user_prompt
        assert "=== 1. REQUIRED DECISION FIELDS ===" not in user_prompt
        assert "=== 7. WHATSAPP AUTHORIZATION ===" not in user_prompt
        assert "=== 8. SITE VISITS AND APPOINTMENTS ===" not in user_prompt
        assert "=== 10. RESPONSE GENERATION ===" not in user_prompt
        assert "=== FINAL RULE ===" not in user_prompt

    def test_user_prompt_preserves_all_dynamic_context(self):
        """Verify dynamic state, slots, pending actions, and caller message remain completely intact."""
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-dedup-2", "+15551234567", domain=DomainType.INFRA_BUILDER)
        state.update_slot("preferred_project", "Pristine City")
        state.update_slot("budget", "1.5 Crores")
        state.set_intent("site_visit_inquiry", clear_pending_question=False)
        state.set_pending_question("Would you like to schedule a site visit?")
        state.set_pending_action("find_available_slots")

        user_prompt = synthesizer.build_user_prompt(state, "Yes, I would like to visit this Saturday.")

        assert "=== PERSISTENT CONVERSATION SLOTS ===" in user_prompt
        assert "preferred_project: Pristine City" in user_prompt
        assert "budget: 1.5 Crores" in user_prompt
        assert "=== CURRENT STATE ===" in user_prompt
        assert "Current intent: site_visit_inquiry" in user_prompt
        assert "Pending question: Would you like to schedule a site visit?" in user_prompt
        assert "Pending action: find_available_slots" in user_prompt
        assert "=== LATEST CALLER MESSAGE" in user_prompt
        assert '"Yes, I would like to visit this Saturday."' in user_prompt

    def test_system_prompt_retains_complete_decision_contract_and_schema(self):
        """Verify system prompt continues to house the complete decision schema, rules, and contract."""
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-dedup-3", "+15551234567", domain=DomainType.INFRA_BUILDER)

        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)

        # Output format and schema contract
        assert "=== OUTPUT FORMAT & CONTRACT ===" in system_prompt
        assert "detected_domain" not in system_prompt
        assert "detected_intent" in system_prompt
        assert "detected_sub_intent" not in system_prompt
        assert "extracted_slots" in system_prompt
        assert "proposed_stage" in system_prompt
        assert "user_facing_response" in system_prompt
        assert "knowledge_required" in system_prompt
        assert "knowledge_query" in system_prompt
        assert "action_proposed" not in system_prompt
        assert "proposed_action" in system_prompt
        assert "needs_clarification" in system_prompt
        assert "clarification_question" not in system_prompt
        assert "suggested_termination" not in system_prompt

        # Core behavioral grounding & action safety
        assert "STRICT GROUNDING & UNTRUSTED DATA" in system_prompt
        assert "WHATSAPP MESSAGING" in system_prompt
        assert "MEETING AND SITE-VISIT INFORMATION" in system_prompt
        assert "EXTERNAL ACTIONS & VERIFICATION" in system_prompt

    def test_user_prompt_size_reduction_magnitude(self):
        """Verify user prompt character count is reduced by over 90% without decision instructions."""
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-dedup-4", "+15551234567", domain=DomainType.INFRA_BUILDER)

        user_prompt = synthesizer.build_user_prompt(state, "What is the price?")
        # Dynamic prompt without 12KB instructions should be under 1,000 characters
        assert len(user_prompt) < 1000
        assert len(user_prompt) > 50  # Contains state and message

    def test_sub_intent_prompt_context_omitted_from_system_and_user_prompts(self):
        """Phase 5D-3G: Verify sub-intent context is removed from prompt construction while state preserves it."""
        synthesizer = PromptSynthesizer()
        mgr = ConversationStateManager()
        state = mgr.create_inbound_state("call-phase5d3g-1", "+15551234567", domain=DomainType.INFRA_BUILDER)
        state.set_intent("property_search", sub_intent="amenities_query")

        # Verify state model still preserves sub_intent
        assert state.current_intent == "property_search"
        assert state.current_sub_intent == "amenities_query"

        # System prompt inspection
        system_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
        assert "Intent: property_search" in system_prompt
        assert "Sub-intent" not in system_prompt
        assert "current_sub_intent" not in system_prompt

        # User prompt inspection
        user_prompt = synthesizer.build_user_prompt(state, "Do you have a gym?")
        assert "Current intent: property_search" in user_prompt
        assert "Current sub-intent" not in user_prompt
        assert "Current stage: " in user_prompt
        assert "Active domain: " in user_prompt
        assert "Pending question: " in user_prompt
        assert "Pending action: " in user_prompt

