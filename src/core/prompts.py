"""Dynamic prompt synthesizer for the unified conversational voice agent.

Constructs modular system instructions and turn contexts incorporating active
domain metadata, caller memory, persistent slot memory, intent preemption,
calendar continuity, and strict grounding rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from src.core.types import CallDirection, DomainType
from src.prompts import load_prompt, render_prompt
from src.state.models import ConversationState

if TYPE_CHECKING:
    from src.domains.base import DomainConfig
    from src.domains.registry import DomainRegistry


# -----------------------------------------------------------------------------
# Base behavioral rules that govern all agent turns across all domains
# -----------------------------------------------------------------------------

STABLE_AGENT_RULES = load_prompt("common/rules.txt")


# -----------------------------------------------------------------------------
# Structured JSON output schema
# -----------------------------------------------------------------------------

JSON_SCHEMA_INSTRUCTION = load_prompt("agent/json_schema.txt")


# -----------------------------------------------------------------------------
# Prompt Synthesizer
# -----------------------------------------------------------------------------

class PromptSynthesizer:
    """Composes dynamic prompt contexts for the LLM based on runtime state."""

    def __init__(
        self,
        registry: Optional[DomainRegistry] = None,
    ) -> None:
        if registry is None:
            from src.domains.registry import get_domain_registry
            registry = get_domain_registry()
        self.registry = registry

    # -------------------------------------------------------------------------
    # System Prompt
    # -------------------------------------------------------------------------

    def build_system_instruction(
        self,
        state: ConversationState,
        domain_config: DomainConfig,
    ) -> str:
        """Synthesize the complete system prompt for the current turn."""

        sections: List[str] = [
            STABLE_AGENT_RULES
        ]

        # ---------------------------------------------------------------------
        # Domain configuration
        # ---------------------------------------------------------------------

        domain_section = render_prompt(
            "agent/domain_section.txt",
            name=domain_config.name,
            domain=domain_config.domain.value,
            description=domain_config.description,
            persona_guidelines=domain_config.persona_guidelines,
            supported_intents=", ".join(domain_config.supported_intents),
            supported_slots=", ".join(
                f"{s.name} ({s.slot_type}: {s.description})"
                for s in domain_config.supported_slots
            ),
        ).strip()

        sections.append(domain_section)

        # ---------------------------------------------------------------------
        # Call direction
        # ---------------------------------------------------------------------

        is_outbound = (
            state.metadata.direction == CallDirection.OUTBOUND
        )

        if is_outbound:
            direction_rules = load_prompt("agent/outbound.txt").strip()
        else:
            direction_rules = load_prompt("agent/inbound.txt").strip()

        # ---------------------------------------------------------------------
        # Caller profile
        # ---------------------------------------------------------------------

        caller = state.caller

        known_details: List[str] = []

        if is_outbound:
            agent_name = state.agent_name
            if agent_name:
                known_details.append(f"Agent Identity: {agent_name} representing {domain_config.name}")
            else:
                known_details.append(f"Agent Identity: Representative of {domain_config.name} (No personal agent name configured; DO NOT invent one)")

            if caller.name:
                known_details.append(f"Contact Name: {caller.name} (NOTE: THIS IS THE PERSON BEING CALLED - NOT YOUR NAME)")
            if caller.phone:
                known_details.append(f"Contact Phone: {caller.phone}")
            if caller.email:
                known_details.append(f"Contact Email: {caller.email}")
            if caller.company:
                known_details.append(f"Contact Company/Institution: {caller.company}")
            if caller.campaign_objective:
                known_details.append(f"Campaign Objective: {caller.campaign_objective}")
            if caller.known_purpose:
                known_details.append(f"Known Purpose: {caller.known_purpose}")
        else:
            if caller.name:
                known_details.append(f"Name: {caller.name}")
            if caller.phone:
                known_details.append(f"Phone: {caller.phone}")
            if caller.email:
                known_details.append(f"Email: {caller.email}")
            if caller.company:
                known_details.append(f"Company/Institution: {caller.company}")
            if caller.campaign_objective:
                known_details.append(f"Campaign Objective: {caller.campaign_objective}")
            if caller.known_purpose:
                known_details.append(f"Known Purpose: {caller.known_purpose}")
            if not caller.name and not caller.email:
                known_details.append("Identity status: Anonymous / Not yet identified")

        known_str = (
            "\n".join(f"- {detail}" for detail in known_details)
            if known_details
            else "- No caller details on record yet."
        )

        # ---------------------------------------------------------------------
        # Persistent slot memory
        # ---------------------------------------------------------------------

        persistent_slots: List[str] = []

        for key, value in state.extracted_slots.items():
            if value is None:
                continue

            value_str = str(value).strip()

            if not value_str:
                continue

            persistent_slots.append(
                f"- {key}: {value_str}"
            )

        persistent_slot_str = (
            "\n".join(persistent_slots)
            if persistent_slots
            else "- No persistent conversation slots collected yet."
        )

        # ---------------------------------------------------------------------
        # Session context
        # ---------------------------------------------------------------------

        session_context = render_prompt(
            "agent/session_context.txt",
            stage=state.stage.value,
            current_intent=state.current_intent or "None",
            current_sub_intent=state.current_sub_intent or "None",
            primary_domain=state.primary_domain.value,
            current_domain=state.current_domain.value,
            domain_locked=str(state.domain_locked).lower(),
            business_status=state.business_status,
            conversation_active=state.conversation_active,
            known_caller_profile=known_str,
            persistent_slots=persistent_slot_str,
        ).strip()

        call_context_section = f"{direction_rules}\n\n{session_context}"

        sections.append(call_context_section)

        # ---------------------------------------------------------------------
        # Pending questions / actions
        # ---------------------------------------------------------------------

        hooks: List[str] = []

        if state.pending_question:
            hooks.append(
                f'Previous pending agent question: "{state.pending_question}" '
                f"(Ignore it if caller asked something else.)"
            )

        if state.pending_action:
            hooks.append(
                f'Action currently in progress: "{state.pending_action}".'
            )

        if state.last_action:
            result_text = (
                f"Outcome: {state.last_tool_result.data}"
                if state.last_tool_result
                else "No verification recorded yet."
            )

            hooks.append(
                f'Last executed action: "{state.last_action}" '
                f"({result_text})."
            )

        if state.intent_history:
            hooks.append(
                f"Recent intent history: "
                f"{', '.join(state.intent_history[-3:])}."
            )

        if hooks:
            sections.append(
                "=== CONVERSATION STATE HOOKS ===\n"
                + "\n".join(f"- {hook}" for hook in hooks)
            )

        # ---------------------------------------------------------------------
        # Output schema
        # ---------------------------------------------------------------------

        sections.append(JSON_SCHEMA_INSTRUCTION)

        return "\n\n".join(sections)

    # -------------------------------------------------------------------------
    # User Prompt
    # -------------------------------------------------------------------------

    def build_user_prompt(
        self,
        state: ConversationState,
        latest_message: str,
    ) -> str:
        """Compose the user prompt including history and persistent memory."""

        prompt_parts: List[str] = []

        # ---------------------------------------------------------------------
        # Recent conversation history
        # ---------------------------------------------------------------------

        recent_turns = state.history[-6:]

        if recent_turns:
            history_lines: List[str] = []

            for turn in recent_turns:
                speaker = (
                    "Caller"
                    if turn.role.value == "caller"
                    else "Agent"
                )

                history_lines.append(
                    f"{speaker}: {turn.content}"
                )

            prompt_parts.append(
                "=== RECENT DIALOGUE HISTORY "
                "(CONTEXT ONLY - NOT THE CURRENT INTENT) ==="
            )

            prompt_parts.append(
                "\n".join(history_lines)
            )

            prompt_parts.append(
                "=========================================================="
            )

        # ---------------------------------------------------------------------
        # Persistent slots
        # ---------------------------------------------------------------------

        if state.extracted_slots:
            slot_lines: List[str] = []

            for key, value in state.extracted_slots.items():
                if value is None:
                    continue

                value_str = str(value).strip()

                if not value_str:
                    continue

                slot_lines.append(
                    f"- {key}: {value_str}"
                )

            if slot_lines:
                prompt_parts.append(
                    "=== PERSISTENT CONVERSATION SLOTS ==="
                )

                prompt_parts.append(
                    "\n".join(slot_lines)
                )

                prompt_parts.append(
                    "=========================================================="
                )

        # ---------------------------------------------------------------------
        # Current intent / stage
        # ---------------------------------------------------------------------

        prompt_parts.append(
            f"""=== CURRENT STATE ===
Current intent: {state.current_intent or "None"}
Current sub-intent: {state.current_sub_intent or "None"}
Current stage: {state.stage.value}
Active domain: {state.current_domain.value}
Pending question: {state.pending_question or "None"}
Pending action: {state.pending_action or "None"}"""
        )

        # ---------------------------------------------------------------------
        # Latest caller message
        # ---------------------------------------------------------------------

        prompt_parts.append(
            f"""=== LATEST CALLER MESSAGE
(PRIMARY SIGNAL FOR CURRENT INTENT) ===
"{latest_message}" """
        )

        # ---------------------------------------------------------------------
        # Decision instructions
        # ---------------------------------------------------------------------

        prompt_parts.append(
            load_prompt("agent/decision_instructions.txt").strip()
        )

        return "\n\n".join(prompt_parts)

    # -------------------------------------------------------------------------
    # Outbound opening prompt
    # -------------------------------------------------------------------------

    def build_outbound_opening_prompt(
        self,
        state: ConversationState,
        domain_config: DomainConfig,
    ) -> str:
        """Compose prompt specifically for the initial outbound opening."""

        caller = state.caller
        contact_name = caller.name or state.contact_name
        agent_name = state.agent_name

        if agent_name:
            agent_identity_block = (
                f"- Agent Name: {agent_name}\n"
                f"- Organization: {domain_config.name}\n"
                f"- Role: Voice Representative for {domain_config.name}"
            )
        else:
            agent_identity_block = (
                f"- Agent Name: None configured (DO NOT invent an agent name; state you are calling from {domain_config.name})\n"
                f"- Organization: {domain_config.name}\n"
                f"- Role: Voice Representative for {domain_config.name}"
            )

        contact_lines: List[str] = []
        if contact_name:
            contact_lines.append(f"Contact Name: {contact_name} (NOTE: This is the person you are calling, NOT your name)")
        else:
            contact_lines.append("Contact Name: Not provided")

        if caller.phone:
            contact_lines.append(f"Contact Phone: {caller.phone}")
        if caller.email:
            contact_lines.append(f"Contact Email: {caller.email}")
        if caller.company:
            contact_lines.append(f"Company/Institution: {caller.company}")
        if caller.campaign_objective:
            contact_lines.append(f"Campaign Objective: {caller.campaign_objective}")
        if caller.known_purpose:
            contact_lines.append(f"Known Purpose: {caller.known_purpose}")

        contact_block = "\n".join(f"- {line}" for line in contact_lines)

        return render_prompt(
            "conversation/outbound_opening.txt",
            domain_name=domain_config.name,
            domain_value=domain_config.domain.value,
            agent_identity_block=agent_identity_block,
            contact_block=contact_block,
            context_block=contact_block,
            contact_name=contact_name or "there",
            agent_name=agent_name or "",
        ).strip()