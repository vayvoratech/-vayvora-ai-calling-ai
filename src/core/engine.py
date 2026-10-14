from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.core.decision import (
    ConversationalDecision,
    DecisionValidator,
    ProposedAction,
    classify_inbound_domain,
    detect_explicit_domain_correction,
    detect_explicit_email_correction,
    extract_datetime_preference,
    extract_email_address,
    is_ambiguous_purpose,
    is_noise_or_unclear,
    is_pure_greeting,
    is_unclear_or_invalid_email_attempt,
    is_valid_email,
)
from src.core.interfaces import KnowledgeProvider, LLMProvider, ToolProvider
from src.core.prompts import PromptSynthesizer
from src.prompts import load_prompt, render_prompt
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
    TurnRole,
)
from src.logging import get_logger
from src.state.models import ConversationState

from src.tools.message_provider import validate_phone_number

if TYPE_CHECKING:
    from src.domains.base import DomainConfig
    from src.domains.registry import DomainRegistry

logger = get_logger("core.engine")


def generate_infra_builder_whatsapp_message(
    project_name: Optional[str] = None,
    rag_context: Optional[str] = None,
    caller_name: Optional[str] = None,
) -> str:
    """Build a concise WhatsApp message without inventing property facts."""
    greeting = f"Hello {caller_name.strip()},\n\n" if caller_name and caller_name.strip() else "Hello,\n\n"
    project = (project_name or "").strip()
    context = (rag_context or "").strip()

    if context:
        # Keep only retrieved content; the knowledge provider is the source of property facts.
        lines = [
            re.sub(r"^[-*#\s]+", "", line).strip()
            for line in context.splitlines()
            if line.strip()
            and not line.lstrip().startswith("<")
            and not line.lstrip().upper().startswith("NOTICE")
            and not line.lstrip().lower().startswith("treat ")
        ]
        summary = "\n".join(f"• {line}" for line in lines[:6])
        heading = f"Here is the verified information for {project}:\n\n" if project else "Here is the verified property information: \n\n"
        body = heading + summary
        body += "\n\nPlease reply here if you would like any other verified details.\n- Infra Builder Team"
        return greeting + body

    # No verified property facts were retrieved. Do not manufacture projects, prices,
    # availability, amenities, or imply that an attachment was sent.
    subject = f" regarding {project}" if project else ""
    return (
        greeting
        + f"Thank you for your interest in Infra Builder{subject}. "
        "I do not have verified property details to include in this message right now. "
        "Please reply here with the project you are interested in, and our team can assist.\n"
        "- Infra Builder Team"
    )


class EngineTurnResult(BaseModel):
    """Encapsulates the outcome of a single conversational turn."""

    model_config = ConfigDict(extra="ignore")

    response_text: str = Field(..., description="Agent utterance voiced to the caller")
    decision: ConversationalDecision = Field(..., description="Structured decision produced by the LLM")
    knowledge_required: bool = Field(default=False, description="Whether domain RAG grounding is needed")
    knowledge_query: Optional[str] = Field(default=None, description="Retrieval query if grounding is needed")
    action_proposed: bool = Field(default=False, description="Whether an external tool action was proposed")
    proposed_action: Optional[ProposedAction] = Field(default=None, description="Details of proposed action")
    conversation_active: bool = Field(default=True, description="Whether conversation remains open")
    termination_occurred: bool = Field(default=False, description="Whether call was formally terminated")
    grounded_citations: List[str] = Field(default_factory=list, description="IDs of knowledge chunks used")
    tool_result: Optional[ToolExecutionResult] = Field(default=None, description="Result of verified tool execution")
    latency_ms: float = Field(default=0.0, description="Total turn execution latency in milliseconds")
    latency_breakdown: Dict[str, float] = Field(default_factory=dict, description="Fine-grained stage timing breakdown in ms")
    raw_llm_response: Optional[str] = Field(default=None, description="Raw LLM provider response")


def is_reusable_first_response(
    response: Optional[str],
    user_query: str,
    formatted_context: str,
    requires_action: bool = False,
) -> bool:
    """Determine whether decision.user_facing_response can be safely reused without a second LLM synthesis.

    Validates all 8 criteria:
    1. Actually user-facing content (non-empty, non-trivial, natural speech).
    2. Sufficiently complete (not a placeholder, stalling, or deferral phrase).
    3. Consistent with the latest caller request.
    4. Grounded in verified RAG result (key facts align, no ungrounded claims).
    5. Does not require waiting for external tool execution results.
    6. Does not contain unresolved templates/placeholders ({name}, [slot], etc.).
    7. Does not leak internal system prompts, JSON, schemas, or tool states.
    8. Satisfies spoken conversation rules (concise, conversational, clear).
    """
    if not response or not isinstance(response, str):
        return False

    resp = response.strip()

    # Rule 1 & 8: Must be user-facing, conversational, reasonable length (>15 and <800 chars)
    if len(resp) < 15 or len(resp) > 800:
        return False

    # Rule 5: Cannot reuse if awaiting tool execution
    if requires_action:
        return False

    resp_lower = resp.lower()

    # Rule 2: Not a stalling / deferral / empty placeholder phrase
    deferral_phrases = [
        "i'll check that for you",
        "i will check that for you",
        "let me check that",
        "let me look into that",
        "let me check our",
        "one moment please",
        "please hold",
        "i am checking",
        "hold on a second",
        "give me a moment",
        "i'll look that up",
        "let me verify that",
        "checking that for you",
        "i'm checking",
        "let me check",
        "checking now",
    ]
    for phrase in deferral_phrases:
        if phrase in resp_lower:
            return False

    # Rule 6: No unresolved placeholders
    placeholder_patterns = [
        "{", "}", "[name]", "[date]", "[time]", "[course]", "[email]",
        "<slot>", "<name>", "<email>", "todo", "tbd", "null", "undefined",
    ]
    for p in placeholder_patterns:
        if p in resp_lower:
            return False

    # Rule 7: No internal system/tool/prompt leakage
    leak_terms = [
        "json", "conversationaldecision", "system_instruction", "tool_name",
        "proposed_action", "redis", "database query", "api response", "user_facing_response",
    ]
    for leak in leak_terms:
        if leak in resp_lower:
            return False

    # Rule 4: Grounding check against formatted_context
    if formatted_context:
        ctx_lower = formatted_context.lower()
        stop_words = {
            "this", "that", "with", "from", "have", "were", "what", "when",
            "where", "which", "your", "about", "there", "their", "would",
            "could", "should", "shall", "these", "those", "also", "into",
            "some", "such", "than", "then", "them", "they", "will", "more",
            "most", "other", "very", "just", "well", "been", "only", "even",
        }
        resp_words = [
            w.strip(".,!?:;\"'()[]")
            for w in resp_lower.split()
            if len(w.strip(".,!?:;\"'()[]")) >= 4 and w.strip(".,!?:;\"'()[]") not in stop_words
        ]

        if not resp_words:
            return False

        matches = sum(1 for w in resp_words if w in ctx_lower)
        overlap_ratio = matches / len(resp_words) if resp_words else 0.0

        # At least 2 key terms or 25% of content terms should be confirmed in context
        if matches < 2 and overlap_ratio < 0.25:
            return False

    return True


class ConversationEngine:
    """Core runtime engine processing turns for the Unified Voice Agent."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        knowledge_provider: Optional[KnowledgeProvider] = None,
        tool_provider: Optional[ToolProvider] = None,
        synthesizer: Optional[PromptSynthesizer] = None,
        registry: Optional[DomainRegistry] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.llm = llm_provider
        self.knowledge_provider = knowledge_provider
        self.tool_provider = tool_provider
        self.settings = settings or get_settings()
        if registry is None:
            from src.domains.registry import get_domain_registry
            registry = get_domain_registry()
        self.registry = registry
        self.synthesizer = synthesizer or PromptSynthesizer(registry=self.registry)
        self.validator = DecisionValidator(registry=self.registry)

    async def start_outbound_conversation(
        self, state: ConversationState
    ) -> EngineTurnResult:
        """Start an outbound call deterministically; no LLM round-trip is needed for identity gating."""
        if state.metadata.direction != CallDirection.OUTBOUND:
            raise ValueError(
                f"Cannot start outbound conversation on an inbound call (call_id: {state.metadata.call_id})"
            )

        t_start = time.perf_counter()
        # This deployment has one active business domain.
        state.lock_domain(DomainType.INFRA_BUILDER)
        domain_config = self.registry.get_or_raise(DomainType.INFRA_BUILDER)

        contact_name = (state.caller.name or state.contact_name or "").strip()
        agent_name = (state.agent_name or (self.settings.agent_name if self.settings else None) or "").strip()
        intro = f"Hi, I'm {agent_name} from {domain_config.name}." if agent_name else f"Hi, I'm calling from {domain_config.name}."
        if contact_name:
            opening_text = f"{intro} Am I speaking with {contact_name}?"
            pending_question = f"Am I speaking with {contact_name}?"
            state.update_slot("outbound_identity_name_pending", False)
        else:
            opening_text = f"{intro} May I know who I'm speaking with?"
            pending_question = "May I know who I'm speaking with?"
            state.update_slot("outbound_identity_name_pending", True)

        state.update_slot("outbound_identity_verified", False)
        state.set_stage(ConversationStage.GREETING)
        state.set_intent("identity_verification", clear_pending_question=False)
        state.set_pending_question(pending_question)
        state.conversation_active = True
        state.record_turn(role=TurnRole.AGENT, content=opening_text)

        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="identity_verification",
            proposed_stage=ConversationStage.GREETING,
            user_facing_response=opening_text,
            action_proposed=False,
            knowledge_required=False,
        )
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info(
            "Outbound conversation %s started; deterministic identity opening in %.2fms",
            state.metadata.call_id,
            elapsed_ms,
        )
        return EngineTurnResult(
            response_text=opening_text,
            decision=decision,
            conversation_active=True,
            termination_occurred=False,
            latency_ms=elapsed_ms,
            latency_breakdown={"opening_generation_ms": elapsed_ms},
        )

    def synthesize_outbound_greeting(self, state: ConversationState) -> str:
        """Produce a deterministic opening utterance for outbound calls without fabricating details."""
        domain_config = self.registry.get_or_raise(state.current_domain)
        return self._generate_deterministic_outbound_opening(state, domain_config)

    def _generate_deterministic_outbound_opening(
        self, state: ConversationState, domain_config: DomainConfig
    ) -> str:
        """Produce a natural, professional opening utterance using known context without fabricating details."""
        caller = state.caller
        contact_name = (caller.name or state.contact_name or "").strip() or None
        company = caller.company.strip() if caller.company else None
        obj = caller.campaign_objective.strip() if caller.campaign_objective else None
        purpose = caller.known_purpose.strip() if caller.known_purpose else None
        agent_name = state.agent_name or (self.settings.agent_name if self.settings else None)

        reason = obj or purpose
        greeting_contact = f"Hi {contact_name}, " if contact_name else "Hello, "

        if agent_name:
            intro = f"I'm {agent_name} from Infra Builder."
        else:
            intro = "I'm calling from Infra Builder."

        if reason:
            r_clean = reason.rstrip(".!? ")
            for prefix in [
                "follow up with buyer who showed interest in ",
                "follow up with client who showed interest in ",
                "follow up regarding inquiry on ",
                "follow up regarding interest in ",
                "follow up regarding ",
                "follow up about ",
                "follow up on ",
                "inquiry about ",
                "shown interest in ",
                "interest in ",
            ]:
                if r_clean.lower().startswith(prefix):
                    r_clean = r_clean[len(prefix) :]
                    break
            return f"{greeting_contact}{intro} I'm following up on your inquiry regarding {r_clean}. Is now a convenient time for a quick conversation?"
        return f"{greeting_contact}{intro} I'm following up on your recent property inquiry with us. Is now a convenient time for a quick conversation?"

    @staticmethod
    def _voice_spell_email(email: str) -> str:
        """Render an email address in a voice-friendly confirmation format."""
        email = (email or "").strip()
        if "@" not in email:
            return email
        local, domain = email.rsplit("@", 1)
        if "." in domain:
            host, tld = domain.rsplit(".", 1)
            domain_text = f"{host} dot {tld}"
        else:
            domain_text = domain
        return f"{'-'.join(local)} at {domain_text}"

    @staticmethod
    def _extract_spoken_name(message: str) -> Optional[str]:
        """Extract a caller-provided name without including correction wording."""
        text = re.sub(r"\s+", " ", (message or "").strip())
        if not text:
            return None

        # Explicit self-identification. Stop before correction clauses or trailing punctuation.
        patterns = [
            r"^(?:my name is|name is|call me)\s+(.+?)"
            r"(?:\s+(?:not|but)\s+.+)?[.!?]?$",
            r"^(?:i am|i'm|im|this is|it's|it is)\s+(.+?)"
            r"(?:\s+(?:not|but)\s+.+)?[.!?]?$",
        ]
        for pattern in patterns:
            m = re.match(pattern, text, flags=re.IGNORECASE)
            if m:
                value = re.sub(r"[.!?,]+$", "", m.group(1)).strip()
                if value and value.lower() not in {"here", "speaking", "not", "correct"}:
                    return value

        # A short name-only response is acceptable, but not an acknowledgement.
        if len(text.split()) <= 3 and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,40}", text):
            if text.lower() not in {
                "yes", "yeah", "yep", "sure", "okay", "ok", "no", "nope",
                "hello", "hi", "hey", "correct", "that's right",
            }:
                return text.strip(" .,!?")
        return None

    @staticmethod
    def _apply_partial_name_correction(current_name: str, message: str) -> Optional[str]:
        """Correct a name when the caller says, for example, 'Pawan, not Powan'."""
        if not current_name or not message:
            return None

        text = re.sub(r"[.!?]+", " ", message.strip())
        text = re.sub(r"\s+", " ", text).strip()
        patterns = [
            r"(?:my name is|name is|call me|it is|it's|its|i am|i'm|im)\s+([A-Za-z][A-Za-z .'-]{0,40}?)\s+not\s+([A-Za-z][A-Za-z .'-]{0,40})$",
            r"([A-Za-z][A-Za-z'-]*)\s+not\s+([A-Za-z][A-Za-z'-]*)$",
            r"not\s+([A-Za-z][A-Za-z'-]*)\s*(?:,|but|rather|instead)\s*([A-Za-z][A-Za-z'-]*)$",
        ]
        for index, pattern in enumerate(patterns):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            if index in (0, 1):
                new_part, old_part = match.group(1).strip(), match.group(2).strip()
            else:
                old_part, new_part = match.group(1).strip(), match.group(2).strip()

            if old_part.lower() not in current_name.lower():
                continue
            corrected = re.sub(
                re.escape(old_part), new_part, current_name, count=1, flags=re.IGNORECASE
            )
            if corrected.strip() and corrected.lower() != current_name.lower():
                return corrected.strip()
        return None

    @staticmethod
    def _normalize_confirmation_text(message: str) -> str:
        cleaned = re.sub(r"[.!?,]+$", "", (message or "").strip().lower()).strip()
        cleaned = re.sub(r"[,]+", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _is_identity_confirmation(message: str) -> bool:
        text = ConversationEngine._normalize_confirmation_text(message)
        return text in {
            "yes", "yeah", "yep", "yes speaking", "speaking", "that's me", "that is me",
            "this is me", "correct", "right", "yes this is me", "yes that's me",
            "yeah speaking", "yep speaking", "this is he", "this is she", "yes correct",
        }

    @staticmethod
    def _is_identity_denial(message: str) -> bool:
        text = (message or "").strip().lower()
        return text in {
            "no", "nope", "not me", "wrong person", "you have the wrong person",
        } or text.startswith("no ") or "wrong number" in text

    @staticmethod
    def _is_person_unavailable(message: str) -> bool:
        text = (message or "").strip().lower()
        return any(p in text for p in [
            "not here", "isn't here", "is not here", "not available", "isn't available",
            "is not available", "away", "out right now", "not around",
        ])

    @staticmethod
    def _apply_partial_email_correction(current_email: str, message: str) -> Optional[str]:
        """Apply spoken local-part corrections while an email confirmation is pending."""
        if not current_email or not message:
            return None

        text = re.sub(r"[.!?]+", " ", message.strip().lower())
        text = re.sub(r"\s+", " ", text).strip()
        local, sep, domain = current_email.partition("@")
        if not sep or not local or not domain:
            return None

        # Handle explicit correction forms, including natural STT variants:
        # "it is pawan not powan", "it's pawan not powan",
        # "pawan not powan", "not powan, pawan", etc.
        patterns = [
            r"(?:it is|it's|its)\s+([a-z0-9._+-]+)\s+not\s+([a-z0-9._+-]+)",
            r"([a-z0-9._+-]+)\s+not\s+([a-z0-9._+-]+)",
            r"not\s+([a-z0-9._+-]+)\s*(?:,|but|rather|instead)\s*([a-z0-9._+-]+)",
            r"([a-z0-9._+-]+)\s*(?:,|but)\s*not\s+([a-z0-9._+-]+)",
        ]

        for index, pattern in enumerate(patterns):
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue

            if index in (0, 1):
                new, old = m.group(1), m.group(2)
            elif index == 2:
                old, new = m.group(1), m.group(2)
            else:
                new, old = m.group(1), m.group(2)

            if old.lower() not in local.lower():
                continue

            corrected_local = re.sub(
                re.escape(old), new, local, count=1, flags=re.IGNORECASE
            )
            candidate = f"{corrected_local}@{domain}"
            if is_valid_email(candidate) and candidate.lower() != current_email.lower():
                return candidate

        return None

    async def _handle_outbound_identity_gate(
        self, state: ConversationState, cleaned_msg: str
    ) -> Optional[EngineTurnResult]:
        """Gate outbound conversation until the intended person is identified."""
        if state.metadata.direction != CallDirection.OUTBOUND:
            return None

        verified = state.get_slot("outbound_identity_verified")
        if verified is True:
            return None

        designated_name = (state.caller.name or state.contact_name or "").strip()
        waiting_for_handover = state.get_slot("outbound_handover_pending") is True
        waiting_for_name = state.get_slot("outbound_identity_name_pending") is True

        response: Optional[str] = None
        should_end = False
        identified_name: Optional[str] = None

        if waiting_for_handover:
            if self._is_person_unavailable(cleaned_msg):
                response = "No problem. I'll call back another time. Thank you."
                should_end = True
            elif self._is_identity_confirmation(cleaned_msg) or (designated_name and designated_name.lower() in cleaned_msg.lower()):
                state.update_slot("outbound_identity_verified", True)
                state.update_slot("outbound_handover_pending", False)
                response = "Thank you. Is this a good time to speak?"
            else:
                response = "Sure, could you please hand the phone to the person I was calling for?"

        elif designated_name:
            if self._is_identity_confirmation(cleaned_msg):
                state.update_slot("outbound_identity_verified", True)
                response = "Great. Is this a good time to speak?"
            elif self._is_identity_denial(cleaned_msg):
                response = f"Could you please hand the phone to {designated_name}?"
                state.update_slot("outbound_handover_pending", True)
            elif self._is_person_unavailable(cleaned_msg):
                response = "No problem. I'll call back another time. Thank you."
                should_end = True
            else:
                identified_name = self._extract_spoken_name(cleaned_msg)
                if identified_name and identified_name.lower() != designated_name.lower():
                    response = f"Could you please hand the phone to {designated_name}?"
                    state.update_slot("outbound_handover_pending", True)
                else:
                    response = f"Am I speaking with {designated_name}?"

        elif waiting_for_name:
            identified_name = self._extract_spoken_name(cleaned_msg)
            if identified_name:
                state.caller.name = identified_name
                state.update_slot("caller_name", identified_name, sync_caller=True)
                state.update_slot("outbound_identity_verified", True)
                state.update_slot("outbound_identity_name_pending", False)
                response = f"Thanks, {identified_name}. Is this a good time to speak?"
            elif self._is_person_unavailable(cleaned_msg):
                response = "No problem. I'll call back another time. Thank you."
                should_end = True
            else:
                response = "Sorry, may I know who I'm speaking with?"

        else:
            # No trusted/designated name: ask the person on the line to identify themselves.
            state.update_slot("outbound_identity_name_pending", True)
            response = "Hello, I'm calling from Infra Builder. May I know who I'm speaking with?"

        if response is None:
            return None

        decision = ConversationalDecision(
            detected_domain=state.current_domain,
            detected_intent="identity_verification",
            proposed_stage=ConversationStage.GREETING if not should_end else ConversationStage.COMPLETED,
            user_facing_response=response,
            action_proposed=False,
            knowledge_required=False,
            suggested_termination=should_end,
        )
        state.record_turn(TurnRole.CALLER, cleaned_msg, intent="identity_verification")
        state.record_turn(TurnRole.AGENT, response)
        if should_end:
            state.request_termination(reason="designated_person_unavailable")
            state.conversation_active = False
        else:
            state.set_stage(ConversationStage.GREETING)
            state.set_intent("identity_verification", clear_pending_question=False)
            state.set_pending_question("Is this a good time to speak?")
        return EngineTurnResult(
            response_text=response,
            decision=decision,
            conversation_active=not should_end,
            termination_occurred=should_end,
        )

    async def process_user_turn(
        self, state: ConversationState, user_message: str, tracker: Optional[Any] = None
    ) -> EngineTurnResult:
        """Process an incoming caller utterance and advance conversation state."""
        cleaned_msg = user_message.strip()

        # Outbound identity must be verified before any RAG or external-action flow.
        identity_result = await self._handle_outbound_identity_gate(state, cleaned_msg)
        if identity_result is not None:
            return identity_result

        # 1. Check for immediate explicit termination from user utterance
        if ConversationState.is_explicit_termination(cleaned_msg):
            logger.info("Explicit termination signal detected from caller: '%s'", cleaned_msg)
            farewell = "Thank you for speaking with us today. Have a great day ahead! Goodbye."
            state.record_turn(TurnRole.CALLER, cleaned_msg)
            state.record_turn(TurnRole.AGENT, farewell)
            state.request_termination(reason=f"caller_explicit_farewell: {cleaned_msg}")

            synthetic_decision = ConversationalDecision(
                detected_domain=state.current_domain,
                detected_intent="end_call",
                proposed_stage=ConversationStage.COMPLETED,
                user_facing_response=farewell,
                suggested_termination=True,
            )
            return EngineTurnResult(
                response_text=farewell,
                decision=synthetic_decision,
                conversation_active=False,
                termination_occurred=True,
            )

        t_start = time.perf_counter()
        timing: Dict[str, float] = {}
        is_inbound = (state.metadata.direction == CallDirection.INBOUND)

        # 2. Check for unclear / STT noise on inbound calls before domain is locked
        if is_inbound and is_noise_or_unclear(cleaned_msg):
            if not state.domain_locked and state.current_domain == DomainType.UNKNOWN:
                unclear_response = "I'm sorry, I didn't quite catch that. Could you please tell me what you're calling about?"
                decision = ConversationalDecision(
                    detected_domain=DomainType.UNKNOWN,
                    detected_intent="clarification",
                    proposed_stage=ConversationStage.PURPOSE_DISCOVERY,
                    user_facing_response=unclear_response,
                    knowledge_required=False,
                    needs_clarification=True,
                    clarification_question="Could you please tell me what you're calling about?",
                )
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, unclear_response)
                return EngineTurnResult(
                    response_text=unclear_response,
                    decision=decision,
                    conversation_active=True,
                    termination_occurred=False,
                )

        # 3. Handle pure greeting on inbound call before domain is locked
        if (
            is_inbound
            and not state.domain_locked
            and state.current_domain == DomainType.UNKNOWN
            and is_pure_greeting(cleaned_msg)
        ):
            greeting_resp = "Hi, how can I help you today?"
            decision = ConversationalDecision(
                detected_domain=DomainType.UNKNOWN,
                detected_intent="greeting",
                proposed_stage=ConversationStage.GREETING,
                user_facing_response=greeting_resp,
                knowledge_required=False,
                action_proposed=False,
            )
            state.set_stage(ConversationStage.GREETING)
            state.set_intent("greeting", clear_pending_question=False)
            state.record_turn(TurnRole.CALLER, cleaned_msg, intent="greeting")
            state.record_turn(TurnRole.AGENT, greeting_resp)
            return EngineTurnResult(
                response_text=greeting_resp,
                decision=decision,
                conversation_active=True,
                termination_occurred=False,
            )

        # 4. Handle ambiguous purpose on inbound call before domain is locked
        if (
            is_inbound
            and not state.domain_locked
            and state.current_domain == DomainType.UNKNOWN
            and is_ambiguous_purpose(cleaned_msg)
        ):
            clarify_resp = "Of course. Are you calling about a property, pricing, or arranging a site visit?"
            decision = ConversationalDecision(
                detected_domain=DomainType.UNKNOWN,
                detected_intent="purpose_discovery",
                proposed_stage=ConversationStage.PURPOSE_DISCOVERY,
                user_facing_response=clarify_resp,
                knowledge_required=False,
                needs_clarification=True,
                clarification_question=clarify_resp,
            )
            state.set_stage(ConversationStage.PURPOSE_DISCOVERY)
            state.set_intent("purpose_discovery", clear_pending_question=False)
            state.set_pending_question(clarify_resp)
            state.record_turn(TurnRole.CALLER, cleaned_msg, intent="purpose_discovery")
            state.record_turn(TurnRole.AGENT, clarify_resp)
            return EngineTurnResult(
                response_text=clarify_resp,
                decision=decision,
                conversation_active=True,
                termination_occurred=False,
            )

        # 5. Inbound Dynamic Domain Pre-classification and Locking / Explicit Correction
        if is_inbound:
            if state.domain_locked:
                # Check for explicit caller correction indicating original domain was wrong
                corrected_domain = detect_explicit_domain_correction(cleaned_msg, state.current_domain)
                if corrected_domain:
                    logger.info(
                        "Explicit caller domain correction accepted: %s -> %s",
                        state.current_domain.value,
                        corrected_domain.value,
                    )
                    state.lock_domain(corrected_domain)
            else:
                # Initial routing: lock directly to Infra Builder
                state.lock_domain(DomainType.INFRA_BUILDER)
                logger.info("Inbound domain resolved and locked to %s", DomainType.INFRA_BUILDER.value)

        # 6. Retrieve active domain configuration
        domain_config = self.registry.get_or_raise(state.current_domain)

        # Infra Builder uses the authorized WhatsApp flow, not the retired course-email workflow.
        if state.current_domain == DomainType.INFRA_BUILDER and state.pending_action == "send_email":
            state.set_pending_action(None)
            state.update_slot("email_confirmation_pending", None)
            state.update_slot("email_confirmed", False)

        # 7. Synthesize prompts
        t_prompt = time.perf_counter()
        system_instruction = self.synthesizer.build_system_instruction(state, domain_config)
        user_prompt = self.synthesizer.build_user_prompt(state, cleaned_msg)
        timing["prompt_construction_ms"] = round((time.perf_counter() - t_prompt) * 1000, 2)

        # 8. Generate structured decision from LLM
        t_llm = time.perf_counter()
        if tracker:
            tracker.mark_decision_start()
        raw_llm_text: Optional[str] = None
        if hasattr(self.llm, "generate_decision"):
            decision: ConversationalDecision = await self.llm.generate_decision(
                user_prompt, system_instruction
            )
        else:
            raw_response = await self.llm.generate_response(user_prompt, system_instruction)
            raw_llm_text = raw_response
            from src.core.llm import extract_json_block
            import json

            data = json.loads(extract_json_block(raw_response))
            decision = ConversationalDecision.model_validate(data)
            decision = self.validator.validate_and_filter(decision)
        if tracker:
            tracker.mark_decision_end()
        timing["llm_decision_ms"] = round((time.perf_counter() - t_llm) * 1000, 2)

        # 9. Domain Lock Enforcement (Post-LLM):
        if is_inbound:
            if state.domain_locked:
                # Application state is AUTHORITATIVE. Overrule any casual domain switch from LLM.
                if decision.detected_domain != state.current_domain:
                    logger.info(
                        "Overriding LLM suggested domain %s to locked domain %s",
                        decision.detected_domain.value,
                        state.current_domain.value,
                    )
                    decision.detected_domain = state.current_domain
            else:
                # Domain was not locked. Lock to Infra Builder.
                state.lock_domain(DomainType.INFRA_BUILDER)
                decision.detected_domain = DomainType.INFRA_BUILDER
                if decision.user_facing_response in ("Hi, how can I help you today?", None, ""):
                    decision.user_facing_response = "Hello! Welcome to Infra Builder. How can I assist you with your property search today?"
        else:
            # Outbound calls: ensure active domain is Infra Builder
            state.current_domain = DomainType.INFRA_BUILDER
            decision.detected_domain = DomainType.INFRA_BUILDER

        # Capture pending question/action prior to intent preemption
        prior_pending_question = state.pending_question
        prior_pending_action = state.pending_action

        # 6. Intent Preemption & Updating
        # The latest explicit user intent has priority over prior pending questions
        if decision.detected_intent:
            state.set_intent(
                new_intent=decision.detected_intent,
                sub_intent=decision.detected_sub_intent,
                clear_pending_question=True,
            )

        # Check if caller message provides or updates date/time preference
        extracted_pref = extract_datetime_preference(cleaned_msg)
        if extracted_pref:
            if not decision.extracted_slots.get("meeting_preference"):
                decision.extracted_slots["meeting_preference"] = extracted_pref
            elif any(w in cleaned_msg.lower() for w in ["actually", "instead", "make it", "change", "rather"]):
                decision.extracted_slots["meeting_preference"] = extracted_pref

        # Check for email extraction, explicit email correction, or unclear email attempts
        explicit_email_corr = detect_explicit_email_correction(cleaned_msg)
        extracted_email = extract_email_address(cleaned_msg)
        email_confirmation_pending = (state.get_slot("email_confirmation_pending") if state.current_domain != DomainType.INFRA_BUILDER else None)

        # Deterministically handle a spoken correction BEFORE LLM intent handling can
        # overwrite the pending email state. This is critical for turns such as
        # "it is pawan not powan" followed by "in email".
        if email_confirmation_pending and not extracted_email and not explicit_email_corr:
            corrected_pending_email = self._apply_partial_email_correction(
                str(email_confirmation_pending), cleaned_msg
            )
            if corrected_pending_email:
                state.update_slot("email", corrected_pending_email, sync_caller=True)
                state.caller.email = corrected_pending_email
                state.update_slot("email_confirmed", False)
                state.update_slot("email_confirmation_pending", corrected_pending_email)
                state.set_pending_action("send_email")
                state.set_pending_question("Is that email address correct?")
                spell = self._voice_spell_email(corrected_pending_email)
                response = f"Got it. Just to confirm, that's {spell}. Is that correct?"
                decision.user_facing_response = response
                decision.action_proposed = False
                decision.proposed_action = None
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, response)
                return EngineTurnResult(
                    response_text=response,
                    decision=decision,
                    action_proposed=False,
                    proposed_action=None,
                    conversation_active=True,
                    termination_occurred=False,
                )

        is_confirming_email = bool(
            email_confirmation_pending
            and (
                prior_pending_action == "send_email"
                or state.pending_action == "send_email"
            )
        )

        last_agent_text = ""
        for t in reversed(state.history):
            if t.role == TurnRole.AGENT:
                last_agent_text = t.content
                break

        # Keep the caller's name in the canonical profile slot. The LLM may emit
        # "name" or "contact_name" instead of "caller_name", so normalize aliases.
        existing_name = (state.caller.name or state.get_slot("caller_name") or "").strip()
        corrected_name = self._apply_partial_name_correction(existing_name, cleaned_msg) if existing_name else None
        name_prompt_pending = any(
            "name" in (question or "").lower()
            for question in (prior_pending_question, state.pending_question, last_agent_text)
        )
        extracted_spoken_name = None
        if corrected_name:
            extracted_spoken_name = corrected_name
        elif name_prompt_pending or re.match(
            r"^(?:my name is|name is|call me|i am|i'm|im|this is|it's|it is)\\b",
            cleaned_msg,
            flags=re.IGNORECASE,
        ):
            extracted_spoken_name = self._extract_spoken_name(cleaned_msg)

        if extracted_spoken_name:
            state.caller.name = extracted_spoken_name
            state.update_slot("caller_name", extracted_spoken_name, sync_caller=True)
            # Keep legacy aliases synchronized if they were produced by the LLM.
            for alias in ("name", "contact_name"):
                if alias in decision.extracted_slots:
                    decision.extracted_slots[alias] = extracted_spoken_name
            decision.extracted_slots["caller_name"] = extracted_spoken_name
            logger.info("Updated caller name from caller utterance: %s", extracted_spoken_name)

        is_whatsapp_context = bool(
            (prior_pending_action == "send_message" or state.pending_action == "send_message")
            or (last_agent_text and "whatsapp" in last_agent_text.lower())
            or (prior_pending_question and "whatsapp" in prior_pending_question.lower())
            or (state.pending_question and "whatsapp" in state.pending_question.lower())
        )
        is_asking_for_email_address = bool(
            state.current_domain != DomainType.INFRA_BUILDER
            and not is_confirming_email
            and not is_whatsapp_context
            and (
                (prior_pending_question and any(w in prior_pending_question.lower() for w in ["email address", "what email", "share your email"]))
                or (state.pending_question and any(w in state.pending_question.lower() for w in ["email address", "what email", "share your email"]))
                or (last_agent_text and any(w in last_agent_text.lower() for w in ["email address", "what email", "share your email"]))
            )
        )
        is_asking_for_topic = bool(
            (prior_pending_question and any(w in prior_pending_question.lower() for w in ["what information", "what would you like me to send", "which course"]))
            or (state.pending_question and any(w in state.pending_question.lower() for w in ["what information", "what would you like me to send", "which course"]))
        )
        is_email_context = state.current_domain != DomainType.INFRA_BUILDER and not is_whatsapp_context and (is_asking_for_email_address or (
            (prior_pending_action == "send_email" or state.pending_action == "send_email") and not is_asking_for_topic
        ))

        # A clear refusal to a pending WhatsApp-permission question cancels only that message.
        last_agent_lower = (last_agent_text or "").lower()
        permission_question_pending = bool(
            state.pending_action == "send_message"
            and any(term in last_agent_lower for term in ("would you like", "may i send", "shall i send", "do you want me to send"))
            and "whatsapp" in last_agent_lower
        )
        normalized_reply = self._normalize_confirmation_text(cleaned_msg)
        if permission_question_pending and normalized_reply in {
            "no", "nope", "no thanks", "no thank you", "not now", "don't", "dont", "please don't",
            "please dont", "do not send it", "don't send it", "dont send it",
        }:
            state.update_slot("whatsapp_consent", False, sync_caller=True)
            state.set_pending_action(None)
            state.set_pending_question(None)
            refusal_response = "No problem. I won't send anything on WhatsApp."
            refusal_decision = ConversationalDecision(
                detected_domain=state.current_domain,
                detected_intent="whatsapp_refusal",
                proposed_stage=ConversationStage.INFORMATION,
                user_facing_response=refusal_response,
                action_proposed=False,
                proposed_action=None,
                knowledge_required=False,
            )
            state.record_turn(TurnRole.CALLER, cleaned_msg, intent="whatsapp_refusal")
            state.record_turn(TurnRole.AGENT, refusal_response)
            return EngineTurnResult(
                response_text=refusal_response,
                decision=refusal_decision,
                conversation_active=True,
                termination_occurred=False,
            )

        # Handle accepting a WhatsApp or information offer from the agent
        is_accepting_whatsapp_offer = bool(
            not is_confirming_email
            and last_agent_text
            and any(w in last_agent_text.lower() for w in ["whatsapp", "brochure", "details", "information", "send"])
            and any(w in last_agent_text.lower() for w in ["would you like", "should i send", "to your whatsapp", "on whatsapp", "by whatsapp"])
            and self._normalize_confirmation_text(cleaned_msg) in {"yes", "yeah", "yep", "sure", "ok", "okay", "certainly", "please", "yes please", "send it", "send them", "send details", "send me"}
        )
        if is_accepting_whatsapp_offer:
            state.update_slot("whatsapp_consent", True)
            target_proj = state.get_slot("preferred_project")
            if state.caller.phone:
                v_ph, norm_ph, _ = validate_phone_number(state.caller.phone)
                if v_ph:
                    ask_resp = f"Sure! I will send the {target_proj or 'project'} brochure and details to your WhatsApp at {norm_ph}."
                    state.set_pending_action("send_message")
                    state.record_turn(TurnRole.CALLER, cleaned_msg)
                    state.record_turn(TurnRole.AGENT, ask_resp)
                    synthetic_decision = ConversationalDecision(
                        detected_domain=state.current_domain,
                        detected_intent="request_brochure",
                        proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                        user_facing_response=ask_resp,
                        action_proposed=True,
                        proposed_action=ProposedAction(
                            tool_name="send_message",
                            arguments={"recipient": norm_ph, "project_name": target_proj, "consent_confirmed": True},
                            confidence=1.0,
                        ),
                    )
                    decision = synthetic_decision
                else:
                    ask_resp = f"Sure. What WhatsApp phone number should I send the {target_proj or 'project'} details to?"
                    state.set_pending_action("send_message")
                    state.set_pending_question(ask_resp)
                    state.record_turn(TurnRole.CALLER, cleaned_msg)
                    state.record_turn(TurnRole.AGENT, ask_resp)
                    synthetic_decision = ConversationalDecision(
                        detected_domain=state.current_domain,
                        detected_intent="request_brochure",
                        proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                        user_facing_response=ask_resp,
                        action_proposed=False,
                        proposed_action=None,
                    )
                    return EngineTurnResult(
                        response_text=ask_resp,
                        decision=synthetic_decision,
                        conversation_active=True,
                        termination_occurred=False,
                    )
            else:
                ask_resp = f"Sure. What WhatsApp phone number should I send the {target_proj or 'project'} details to?"
                state.set_pending_action("send_message")
                state.set_pending_question(ask_resp)
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, ask_resp)
                synthetic_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="request_brochure",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response=ask_resp,
                    action_proposed=False,
                    proposed_action=None,
                )
                return EngineTurnResult(
                    response_text=ask_resp,
                    decision=synthetic_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )

        # Handle unclear or invalid email attempt
        if state.current_domain != DomainType.INFRA_BUILDER and not is_whatsapp_context and (is_asking_for_email_address or is_unclear_or_invalid_email_attempt(cleaned_msg)) and not extracted_email and not explicit_email_corr:
            if is_unclear_or_invalid_email_attempt(cleaned_msg) or (
                is_asking_for_email_address
                and not any(w in cleaned_msg.lower() for w in ["cancel", "no", "never mind", "stop", "don't", "dont", "what", "why", "how"])
            ):
                unclear_email_response = "Could you repeat the email address for me?"
                state.set_pending_action("send_email")
                state.set_pending_question("Could you repeat the email address for me?")
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, unclear_email_response)
                synthetic_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="provide_email",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response=unclear_email_response,
                    action_proposed=False,
                    proposed_action=None,
                )
                return EngineTurnResult(
                    response_text=unclear_email_response,
                    decision=synthetic_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )

        if explicit_email_corr:
            decision.extracted_slots["email"] = explicit_email_corr
            state.update_slot("email", explicit_email_corr, sync_caller=True)
            state.caller.email = explicit_email_corr
        elif extracted_email:
            decision.extracted_slots["email"] = extracted_email
            state.update_slot("email", extracted_email, sync_caller=True)
            state.caller.email = extracted_email

        # Extract spoken phone number for WhatsApp
        digits_match = re.search(r"\b(?:\+?91[\-\s]?)?\d{5,12}\b", cleaned_msg)
        if digits_match:
            candidate_raw = digits_match.group(0)
            v_ph, n_ph, phone_err = validate_phone_number(candidate_raw)
            if v_ph:
                state.caller.phone = n_ph
                state.update_slot("phone", n_ph, sync_caller=True)
                state.update_slot("whatsapp_number", n_ph)
                # A phone number is destination data, not consent to message it.
                if (
                    (state.pending_action == "send_message" or prior_pending_action == "send_message")
                    and not any(
                        phrase in cleaned_msg.lower()
                        for phrase in (
                            "don't send whatsapp", "dont send whatsapp", "do not send whatsapp",
                            "no whatsapp", "i don't use whatsapp", "i dont use whatsapp",
                            "do not message", "don't message", "dont message",
                            "opt out", "stop messaging", "unsubscribe", "no messages",
                        )
                    )
                ):
                    if state.get_slot("whatsapp_consent") is True:
                        target_proj = state.get_slot("preferred_project")
                        decision.action_proposed = True
                        decision.proposed_action = ProposedAction(
                            tool_name="send_message",
                            arguments={"recipient": n_ph, "project_name": target_proj, "consent_confirmed": True},
                            confidence=1.0,
                        )
                    else:
                        permission_question = f"Thanks. May I send the requested property information to your WhatsApp at {n_ph}?"
                        state.set_pending_action("send_message")
                        state.set_pending_question(permission_question)
                        state.record_turn(TurnRole.CALLER, cleaned_msg)
                        state.record_turn(TurnRole.AGENT, permission_question)
                        permission_decision = ConversationalDecision(
                            detected_domain=state.current_domain,
                            detected_intent="whatsapp_consent",
                            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                            user_facing_response=permission_question,
                            action_proposed=False,
                            proposed_action=None,
                            knowledge_required=False,
                        )
                        return EngineTurnResult(
                            response_text=permission_question,
                            decision=permission_decision,
                            conversation_active=True,
                            termination_occurred=False,
                        )
            elif is_whatsapp_context:
                invalid_phone_resp = f"That phone number appears to be invalid or incomplete ({phone_err}). Could you please share a valid 10-digit WhatsApp number?"
                state.set_pending_action("send_message")
                state.set_pending_question("Could you please share a valid 10-digit WhatsApp number?")
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, invalid_phone_resp)
                synthetic_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="provide_phone",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response=invalid_phone_resp,
                    action_proposed=False,
                    proposed_action=None,
                )
                return EngineTurnResult(
                    response_text=invalid_phone_resp,
                    decision=synthetic_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )

        # 7. Slot Updates
        for slot_key, slot_val in decision.extracted_slots.items():
            state.update_slot(slot_key, slot_val, sync_caller=True)

        if extracted_pref and (not state.get_slot("meeting_preference") or any(w in cleaned_msg.lower() for w in ["actually", "instead", "make it", "change", "rather"])):
            state.update_slot("meeting_preference", extracted_pref, sync_caller=True)

        # 8. Stage Progression
        # Guard: LLM suggestion of completion requires explicit termination check
        if decision.proposed_stage == ConversationStage.COMPLETED:
            if ConversationState.is_explicit_termination(cleaned_msg):
                state.request_termination(reason="confirmed_completion_at_closing")
            else:
                # Do NOT terminate on non-explicit cues (e.g. saying "no")
                state.set_stage(ConversationStage.INFORMATION)
        else:
            state.set_stage(decision.proposed_stage)

        # 9. Handle Proposed Action
        # CRITICAL: We distinguish proposed action from executed action.
        # We record pending_action, but DO NOT claim tool success.
        action_proposed = decision.action_proposed and decision.proposed_action is not None

        # If the caller chooses one of the actual returned slots, confirm that exact slot before booking.
        available_slots_for_selection = state.get_slot("available_slots") or []
        if prior_pending_action == "create_calendar_event" and available_slots_for_selection:
            def _slot_key(value: Any) -> str:
                value = str(value or "").lower().strip()
                value = re.sub(r"(?<=\d):00(?=\s*(?:am|pm)\b)", "", value)
                return re.sub(r"[^a-z0-9]", "", value)

            caller_slot_key = _slot_key(cleaned_msg)
            selected_slot = next(
                (
                    str(candidate)
                    for candidate in available_slots_for_selection
                    if _slot_key(candidate)
                    and (
                        _slot_key(candidate) in caller_slot_key
                        or caller_slot_key in _slot_key(candidate)
                    )
                    and len(caller_slot_key) >= 3
                ),
                None,
            )
            if selected_slot:
                state.update_slot("meeting_preference", selected_slot, sync_caller=True)
                state.update_slot("visit_confirmation_pending", selected_slot)
                state.update_slot("visit_confirmed", False)
                response = f"Would you like me to book the site visit for {selected_slot}?"
                state.set_pending_action("create_calendar_event")
                state.set_pending_question(response)
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, response)
                slot_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="site_visit_request",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response=response,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                return EngineTurnResult(
                    response_text=response,
                    decision=slot_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )

        # Promote to calendar action if caller is requesting scheduling or answering pending date/time question
        is_calendar_request = any(
            w in cleaned_msg.lower()
            for w in ["schedule", "consultation", "demo", "calendar", "book a slot", "site visit", "visit", "tour"]
        )
        is_explicit_confirmation = self._normalize_confirmation_text(cleaned_msg) in {
            "yes", "yeah", "yep", "sure", "confirm", "please book", "book it", "that works", "sounds good",
            "yes please", "correct", "perfect"
        } or any(cue in cleaned_msg.lower() for cue in ["confirm", "book it", "please book", "that works", "works for me"])

        if not action_proposed and (
            is_calendar_request
            or prior_pending_action in ("create_calendar_event", "find_available_slots")
            or state.pending_action in ("create_calendar_event", "find_available_slots")
            or (
                prior_pending_question
                and any(w in prior_pending_question.lower() for w in ["date", "time slot", "time would", "work best", "schedule", "site visit"])
            )
            or (
                state.pending_question
                and any(w in state.pending_question.lower() for w in ["date", "time slot", "time would", "work best", "schedule", "site visit"])
            )
        ) and (extracted_pref or state.get_slot("meeting_preference")):
            slot_to_book = extracted_pref or state.get_slot("meeting_preference")
            proj_name = state.get_slot("preferred_project") or (decision.proposed_action.arguments.get("property_name") if decision.proposed_action else None)
            # Invariant: confirmed is set to True ONLY if caller explicitly confirmed
            if is_explicit_confirmation:
                if prior_pending_action == "create_calendar_event":
                    offered_slot = state.get_slot("visit_confirmation_pending")
                    if offered_slot:
                        slot_to_book = offered_slot
                    state.update_slot("visit_confirmed", True)
                decision.action_proposed = True
                decision.proposed_action = ProposedAction(
                    tool_name="create_calendar_event",
                    arguments={"slot": slot_to_book, "property_name": proj_name, "confirmed": True},
                )
                action_proposed = True
            else:
                decision.action_proposed = True
                decision.proposed_action = ProposedAction(
                    tool_name="find_available_slots",
                    arguments={"preferred_date": slot_to_book, "property_name": proj_name},
                )
                action_proposed = True

        # Project identity must come from caller-stated/extracted data or trusted state.
        # Do not infer a specific project from generic words such as "villa" or a location alone.

        # WhatsApp sharing flow: asking for permission is a conversational step;
        # the execution guard below independently enforces consent.
        msg_lower = cleaned_msg.lower()
        is_whatsapp_request = any(
            phrase in msg_lower
            for phrase in (
                "whatsapp", "send brochure", "send me the brochure", "send details",
                "send me details", "send the details", "send pricing", "send me pricing",
                "share brochure", "share details", "send it on whatsapp", "share on whatsapp",
            )
        )
        is_whatsapp_opt_out = any(
            phrase in msg_lower
            for phrase in (
                "don't send whatsapp", "dont send whatsapp", "do not send whatsapp",
                "no whatsapp", "i don't use whatsapp", "i dont use whatsapp",
                "do not message", "don't message", "dont message", "opt out",
                "stop messaging", "unsubscribe", "no messages",
            )
        )

        if is_whatsapp_opt_out:
            state.update_slot("whatsapp_consent", False, sync_caller=True)
            decision.extracted_slots["whatsapp_consent"] = False
            if action_proposed and decision.proposed_action and decision.proposed_action.tool_name == "send_message":
                decision.action_proposed = False
                decision.proposed_action = None
                action_proposed = False
            state.set_pending_action(None)
            state.set_pending_question(None)

        caller_phone = state.caller.phone or state.get_slot("phone") or state.get_slot("whatsapp_number")
        if not caller_phone and is_whatsapp_request:
            # If the caller included a number in the same request, retain it without treating it as consent.
            phone_match = re.search(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b", cleaned_msg)
            if phone_match:
                valid, normalized, _ = validate_phone_number(phone_match.group(0))
                if valid:
                    caller_phone = normalized
                    state.caller.phone = normalized
                    state.update_slot("phone", normalized, sync_caller=True)
                    state.update_slot("whatsapp_number", normalized)

        if is_whatsapp_request and not is_whatsapp_opt_out:
            consent = state.get_slot("whatsapp_consent")
            if consent is not True:
                number_suffix = f" at {caller_phone}" if caller_phone else ""
                auth_question = f"Would you like me to send the requested property information to your WhatsApp{number_suffix}?"
                state.set_pending_action("send_message")
                state.set_pending_question(auth_question)
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, auth_question)
                consent_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="whatsapp_consent",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response=auth_question,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                return EngineTurnResult(
                    response_text=auth_question,
                    decision=consent_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )
            if not caller_phone:
                ask_phone = "Which WhatsApp number should I send the requested property information to?"
                state.set_pending_action("send_message")
                state.set_pending_question(ask_phone)
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, ask_phone)
                ask_phone_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="whatsapp_information_request",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response=ask_phone,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                return EngineTurnResult(
                    response_text=ask_phone,
                    decision=ask_phone_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )
            decision.action_proposed = True
            decision.proposed_action = ProposedAction(
                tool_name="send_message",
                arguments={
                    "recipient": str(caller_phone),
                    "project_name": state.get_slot("preferred_project"),
                    "consent_confirmed": True,
                },
                confidence=1.0,
            )
            action_proposed = True

        if action_proposed and decision.proposed_action:
            state.set_pending_action(decision.proposed_action.tool_name)

        # Continue a pending email action after the caller confirms the address, even if
        # the LLM's current turn does not independently propose the tool.
        if (
            not action_proposed
            and prior_pending_action == "send_email"
            and state.get_slot("email_confirmation_pending")
            and self._normalize_confirmation_text(cleaned_msg) in {"yes", "yeah", "yep", "correct", "that's correct", "that is correct", "yes that's correct", "yes that is correct"}
        ):
            confirmed = str(state.get_slot("email_confirmation_pending"))
            decision.action_proposed = True
            decision.proposed_action = ProposedAction(
                tool_name="send_email",
                arguments={"subject": "Course Details & Curriculum", "recipient": confirmed},
            )
            action_proposed = True

        # Email safety gate: never send until the recipient has been explicitly confirmed.
        if state.current_domain != DomainType.INFRA_BUILDER and action_proposed and decision.proposed_action and decision.proposed_action.tool_name == "send_email":
            candidate_email = (
                explicit_email_corr
                or extracted_email
                or (state.caller.email if getattr(state, "contact_email", None) and state.caller.email else None)
            )
            confirmed_email = state.get_slot("email_confirmed")
            pending_email = state.get_slot("email_confirmation_pending")

            # A partial correction such as "not powan, pawan" updates the current candidate.
            if pending_email:
                corrected = self._apply_partial_email_correction(str(pending_email), cleaned_msg)
                if corrected:
                    candidate_email = corrected
                    decision.proposed_action.arguments["recipient"] = corrected
                    state.update_slot("email", corrected, sync_caller=True)
                    state.caller.email = corrected
                    state.update_slot("email_confirmed", False)
                    state.update_slot("email_confirmation_pending", corrected)
                    spell = self._voice_spell_email(corrected)
                    response = f"Got it. Just to confirm, that's {spell}. Is that correct?"
                    decision.user_facing_response = response
                    decision.action_proposed = False
                    decision.proposed_action = None
                    action_proposed = False
                    state.set_pending_action("send_email")
                    state.set_pending_question("Is that email address correct?")
                    state.record_turn(TurnRole.CALLER, cleaned_msg)
                    state.record_turn(TurnRole.AGENT, response)
                    return EngineTurnResult(
                        response_text=response,
                        decision=decision,
                        action_proposed=False,
                        proposed_action=None,
                        conversation_active=True,
                        termination_occurred=False,
                    )

                if self._is_identity_confirmation(cleaned_msg) or self._normalize_confirmation_text(cleaned_msg) in {"yes", "correct", "that's correct", "that is correct"}:
                    confirmed_email = str(pending_email)
                    state.update_slot("email_confirmed", True)
                    state.update_slot("email_confirmation_pending", None)
                    candidate_email = confirmed_email
                    decision.action_proposed = True
                    decision.proposed_action = ProposedAction(
                        tool_name="send_email",
                        arguments={"subject": "Course Details & Curriculum", "recipient": confirmed_email},
                    )
                    action_proposed = True
                elif self._normalize_confirmation_text(cleaned_msg) in {"no", "nope", "not correct", "that's not correct", "that is not correct"}:
                    state.update_slot("email_confirmed", False)
                    response = "No problem. What email address should I use instead?"
                    decision.user_facing_response = response
                    decision.action_proposed = False
                    decision.proposed_action = None
                    action_proposed = False
                    state.set_pending_action("send_email")
                    state.set_pending_question("What email address should I use instead?")
                    state.record_turn(TurnRole.CALLER, cleaned_msg)
                    state.record_turn(TurnRole.AGENT, response)
                    return EngineTurnResult(
                        response_text=response,
                        decision=decision,
                        action_proposed=False,
                        proposed_action=None,
                        conversation_active=True,
                        termination_occurred=False,
                    )

            # If no recipient exists, let the existing deterministic guard ask for it.
            if candidate_email and is_valid_email(str(candidate_email)) and not state.get_slot("email_confirmed"):
                candidate_email = str(candidate_email)
                state.update_slot("email", candidate_email, sync_caller=True)
                state.caller.email = candidate_email
                state.update_slot("email_confirmation_pending", candidate_email)
                state.update_slot("email_confirmed", False)
                spell = self._voice_spell_email(candidate_email)
                response = f"Just to confirm, that's {spell}. Is that correct?"
                decision.user_facing_response = response
                decision.action_proposed = False
                decision.proposed_action = None
                action_proposed = False
                state.set_pending_action("send_email")
                state.set_pending_question("Is that email address correct?")
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, response)
                return EngineTurnResult(
                    response_text=response,
                    decision=decision,
                    action_proposed=False,
                    proposed_action=None,
                    conversation_active=True,
                    termination_occurred=False,
                )

            if state.get_slot("email_confirmed") and confirmed_email:
                decision.proposed_action.arguments["recipient"] = confirmed_email

        # 10. Handle Grounded Knowledge Retrieval (Phase 4)
        grounded_citations: List[str] = []
        final_response_text = decision.user_facing_response

        # Before domain selection: NEVER perform domain-specific RAG.
        if state.current_domain == DomainType.UNKNOWN:
            decision.knowledge_required = False
            decision.knowledge_query = None

        # Simple conversational acknowledgments do not require factual RAG retrieval
        simple_acks = {
            "hello", "hi", "hey", "yes", "yeah", "yep", "sure", "okay", "ok",
            "thanks", "thank you", "great", "fine", "alright", "bye", "goodbye",
            "sounds good", "perfect", "no problem", "correct",
        }
        if cleaned_msg.lower().strip() in simple_acks:
            decision.knowledge_required = False
            decision.knowledge_query = None

        if decision.knowledge_required and not self.knowledge_provider:
            logger.warning("Knowledge was required but no knowledge provider is configured.")
            final_response_text = (
                "I don't have verified property details available right now. "
                "I can help arrange a follow-up or assist with another question."
            )
            decision.user_facing_response = final_response_text
            decision.knowledge_required = False
            decision.knowledge_query = None

        if decision.knowledge_required and self.knowledge_provider:
            t_rag = time.perf_counter()
            if tracker:
                tracker.mark_rag_start()
            query_text = decision.knowledge_query or decision.detected_intent or cleaned_msg
            rag_query = RAGQuery(
                domain=state.current_domain,
                query_text=query_text,
                top_k=self.settings.rag_top_k,
                relevance_threshold=self.settings.rag_relevance_threshold,
            )

            try:
                if hasattr(self.knowledge_provider, "retrieve_grounded_context"):
                    grounded_res = await self.knowledge_provider.retrieve_grounded_context(rag_query)
                else:
                    chunks = await self.knowledge_provider.search(rag_query)
                    from src.rag.retriever import GroundedContextResult

                    grounded_res = GroundedContextResult(
                        query=rag_query,
                        chunks=chunks,
                        knowledge_available=bool(chunks),
                        formatted_context="\n".join(c.content for c in chunks),
                    )
                timing["rag_retrieval_ms"] = round((time.perf_counter() - t_rag) * 1000, 2)
                if tracker:
                    tracker.mark_rag_end()

                if grounded_res.service_unavailable:
                    logger.warning("RAG service unavailable for domain %s", decision.detected_domain.value)
                    final_response_text = (
                        "Our knowledge base is currently undergoing maintenance, so I don't have those specific details on hand right now. "
                        "I can have our team follow up with you directly."
                    )
                    decision.user_facing_response = final_response_text
                elif not grounded_res.knowledge_available or not grounded_res.chunks:
                    logger.info("No verified knowledge passed relevance threshold for query '%s'", query_text)
                    final_response_text = (
                        "I checked our knowledge base, but I don't have verified details regarding that at the moment. "
                        "Would you like me to have our team follow up with you directly, or is there another question I can answer?"
                    )
                    decision.user_facing_response = final_response_text
                else:
                    # Chunks found and passed relevance threshold
                    grounded_citations = [c.doc_id for c in grounded_res.chunks]

                    # PHASE 5: Check whether decision.user_facing_response can be safely reused
                    # to skip the redundant second Gemini round-trip!
                    if is_reusable_first_response(
                        response=decision.user_facing_response,
                        user_query=cleaned_msg,
                        formatted_context=grounded_res.formatted_context,
                        requires_action=bool(action_proposed and decision.proposed_action),
                    ):
                        logger.info("Phase 5: Reusing grounded first LLM response, avoiding second Gemini call.")
                        final_response_text = decision.user_facing_response.strip()
                        if tracker:
                            tracker.reused_first_response = True
                    else:
                        grounding_prompt = render_prompt(
                            "rag/grounded_answer.txt",
                            caller_message=cleaned_msg,
                            domain=decision.detected_domain.value,
                            formatted_context=grounded_res.formatted_context,
                        ).strip()

                        t_ground = time.perf_counter()
                        grounded_speech = await self.llm.generate_response(
                            prompt=grounding_prompt,
                            system_instruction=load_prompt("rag/system.txt").strip(),
                        )
                        timing["grounded_llm_ms"] = round((time.perf_counter() - t_ground) * 1000, 2)
                        final_response_text = grounded_speech.strip()
                        decision.user_facing_response = final_response_text
            except Exception as rag_err:
                timing["rag_retrieval_ms"] = round((time.perf_counter() - t_rag) * 1000, 2)
                if tracker:
                    tracker.mark_rag_end()
                logger.error("RAG retrieval failed: %s", rag_err)
                final_response_text = (
                    "Our knowledge base is currently undergoing maintenance, so I don't have those specific details on hand right now. "
                    "I can have our team follow up with you directly."
                )
                decision.user_facing_response = final_response_text

        # 11. Handle External Action Execution & Verification (Phase 5)
        tool_exec_result: Optional[ToolExecutionResult] = None
        if action_proposed and decision.proposed_action and not self.tool_provider:
            action_name = decision.proposed_action.tool_name
            action_proposed = False
            decision.action_proposed = False
            decision.proposed_action = None
            state.set_pending_action(None)
            final_response_text = (
                "I can't complete that request automatically right now because the required service is unavailable. "
                "I can still answer questions or help you with the next step."
            )
            decision.user_facing_response = final_response_text
            logger.warning("Action %s was not executed: no tool provider configured", action_name)
        if action_proposed and decision.proposed_action and self.tool_provider:
            tool_name = decision.proposed_action.tool_name
            tool_args = dict(decision.proposed_action.arguments)

            # Prohibited placeholders and sender addresses that must never be used as recipient
            prohibited_vals = {
                "default caller",
                "caller@mail",
                "unknown@example.com",
                "john doe",
                "jane doe",
                "default@example.com",
                "fallback@example.com",
                "noreply@infrabuilder.com",
                "info@infrabuilder.com",
                "admin@infrabuilder.com",
                "0000000000",
                "9999999999",
                "1234567890",
                "9876543210",
                "none",
                "null",
                "empty",
            }

            # Strip any hallucinated placeholders from tool arguments
            for arg_k in ["caller_name", "candidate_name", "name", "recipient", "phone", "email", "caller_email"]:
                if arg_k in tool_args and str(tool_args[arg_k]).strip().lower() in prohibited_vals:
                    tool_args.pop(arg_k, None)

            # Auto-populate caller profile details into arguments if absent
            if "caller_name" not in tool_args and state.caller.name and state.caller.name.strip().lower() not in prohibited_vals:
                tool_args["caller_name"] = state.caller.name
            if "caller_phone" not in tool_args and state.caller.phone and state.caller.phone.strip().lower() not in prohibited_vals:
                tool_args["caller_phone"] = state.caller.phone
            if "domain" not in tool_args:
                tool_args["domain"] = state.current_domain.value

            execute_action = True

            allowed_tool_names = {
                "find_available_slots",
                "create_calendar_event",
                "update_lead",
                "send_message",
                "update_business_status",
            }
            if state.current_domain == DomainType.INFRA_BUILDER and tool_name not in allowed_tool_names:
                execute_action = False
                action_proposed = False
                decision.action_proposed = False
                decision.proposed_action = None
                state.set_pending_action(None)
                final_response_text = (
                    "I can't perform that action in this property-assistance workflow. "
                    "I can help with verified property information or a site visit."
                )
                decision.user_facing_response = final_response_text

            # Action 1: send_message (WhatsApp)
            if tool_name == "send_message":
                recipient = (
                    tool_args.get("recipient")
                    or tool_args.get("phone")
                    or state.caller.phone
                    or state.get_slot("phone")
                    or state.get_slot("whatsapp_number")
                    or ""
                )
                if str(recipient).strip().lower() in prohibited_vals:
                    recipient = ""

                if not recipient:
                    phone_match = re.search(r"\b(?:\+?91[\-\s]?)?[6-9]\d{9}\b", cleaned_msg)
                    if phone_match:
                        recipient = phone_match.group(0)

                valid_phone, norm_phone, phone_err = validate_phone_number(str(recipient)) if recipient else (False, "", "Missing phone")

                # Check opt-out
                opt_out = any(phrase in cleaned_msg.lower() for phrase in ["no whatsapp", "don't message", "dont message", "stop messaging", "opt out", "unsubscribe", "no message"])
                if opt_out:
                    execute_action = False
                    tool_args.pop("recipient", None)
                    final_response_text = "Understood. I will not send any WhatsApp messages to your number."
                    decision.user_facing_response = final_response_text
                    state.set_pending_action(None)
                    state.set_pending_question(None)
                elif not valid_phone:
                    execute_action = False
                    tool_args.pop("recipient", None)
                    final_response_text = "Sure, I can share the project brochure and details. What is your WhatsApp number?"
                    decision.user_facing_response = final_response_text
                    state.set_pending_action("send_message")
                    state.set_pending_question("What is your WhatsApp number?")
                else:
                    # Check consent
                    # Only trusted session state set by the explicit consent handler authorizes sending.
                    # Tool arguments and keyword matches are not proof of consent.
                    consent_confirmed = state.get_slot("whatsapp_consent") is True
                    if not consent_confirmed:
                        execute_action = False
                        final_response_text = "Would you like me to send the project brochure and pricing to your WhatsApp?"
                        decision.user_facing_response = final_response_text
                        state.set_pending_action("send_message")
                        state.set_pending_question("Would you like me to send the project brochure and pricing to your WhatsApp?")
                    else:
                        tool_args["recipient"] = norm_phone
                        tool_args["consent_confirmed"] = True

                        proj_name = tool_args.get("project_name") or state.get_slot("preferred_project")

                        rag_ctx = ""
                        if self.knowledge_provider:
                            try:
                                rag_query = RAGQuery(
                                    domain=DomainType.INFRA_BUILDER,
                                    query_text=proj_name or cleaned_msg,
                                    top_k=self.settings.rag_top_k if self.settings else 3,
                                    relevance_threshold=self.settings.rag_relevance_threshold if self.settings else 0.4,
                                )
                                if hasattr(self.knowledge_provider, "retrieve_grounded_context"):
                                    rag_res = await self.knowledge_provider.retrieve_grounded_context(rag_query)
                                    if rag_res and rag_res.formatted_context:
                                        rag_ctx = rag_res.formatted_context
                                else:
                                    chunks = await self.knowledge_provider.search(rag_query)
                                    if chunks:
                                        rag_ctx = "\n".join(c.content for c in chunks)
                            except Exception as err:
                                logger.warning("RAG retrieval for WhatsApp message failed: %s", err)

                        if not rag_ctx.strip():
                            execute_action = False
                            final_response_text = (
                                "I couldn't retrieve verified property information, so I haven't sent a WhatsApp message. "
                                "Would you like me to try again or arrange a team follow-up?"
                            )
                            decision.user_facing_response = final_response_text
                            state.set_pending_action(None)
                            state.set_pending_question(None)
                        else:
                            tool_args["project_name"] = proj_name or "Infra Builder"
                            tool_args["message"] = generate_infra_builder_whatsapp_message(
                                project_name=proj_name,
                                rag_context=rag_ctx,
                                caller_name=state.caller.name,
                            )

            # Action 2: find_available_slots
            elif tool_name == "find_available_slots":
                prop_name = (
                    tool_args.get("property_name")
                    or tool_args.get("project_name")
                    or state.get_slot("preferred_project")
                )
                pref_date = tool_args.get("preferred_date") or tool_args.get("date") or state.get_slot("meeting_preference")
                if not prop_name or not pref_date:
                    execute_action = False
                    missing = "which property or project" if not prop_name else "which date you prefer"
                    final_response_text = f"I can check actual site-visit availability. Could you tell me {missing}?"
                    decision.user_facing_response = final_response_text
                    state.set_pending_action("find_available_slots")
                    state.set_pending_question(final_response_text)
                else:
                    tool_args["property_name"] = prop_name
                    tool_args["preferred_date"] = pref_date

            # Action 3: create_calendar_event
            elif tool_name == "create_calendar_event":
                slot = (
                    tool_args.get("slot")
                    or tool_args.get("start_time")
                    or tool_args.get("time")
                    or state.get_slot("meeting_preference")
                )
                is_confirmed = (
                    tool_args.get("confirmed") is True
                    and state.get_slot("visit_confirmation_pending") == slot
                    and is_explicit_confirmation
                )
                available_slots = state.get_slot("available_slots") or []
                slot_is_available = any(
                    str(candidate).strip().casefold() == str(slot).strip().casefold()
                    for candidate in available_slots
                ) if slot else False
                if not slot:
                    execute_action = False
                    action_proposed = False
                    decision.action_proposed = False
                    decision.proposed_action = None
                    final_response_text = "I'd be happy to arrange a site visit. Which project and date would you prefer?"
                    decision.user_facing_response = final_response_text
                    state.set_pending_action("find_available_slots")
                    state.set_pending_question(final_response_text)
                elif not slot_is_available:
                    # Never create a booking for a slot that was not returned by the availability tool.
                    property_name = tool_args.get("property_name") or state.get_slot("preferred_project")
                    if not property_name:
                        execute_action = False
                        action_proposed = False
                        decision.action_proposed = False
                        decision.proposed_action = None
                        final_response_text = "Which property or project would you like to visit?"
                        decision.user_facing_response = final_response_text
                        state.set_pending_action("find_available_slots")
                        state.set_pending_question(final_response_text)
                    else:
                        tool_name = "find_available_slots"
                        tool_args = {
                            "preferred_date": state.get_slot("meeting_preference") or slot,
                            "property_name": property_name,
                            "domain": DomainType.INFRA_BUILDER.value,
                        }
                        execute_action = True
                        decision.action_proposed = True
                        decision.proposed_action = ProposedAction(
                            tool_name="find_available_slots",
                            arguments=dict(tool_args),
                        )
                        action_proposed = True
                        final_response_text = "I'll check the actual availability for that date before confirming a visit."
                        decision.user_facing_response = final_response_text
                        state.set_pending_action("find_available_slots")
                elif not is_confirmed:
                    execute_action = False
                    final_response_text = f"I found that slot in the availability results: {slot}. Would you like me to book it?"
                    decision.user_facing_response = final_response_text
                    state.set_pending_action("create_calendar_event")
                    state.update_slot("visit_confirmation_pending", slot)
                    state.set_pending_question(f"Would you like me to book the site visit for {slot}?")
                else:
                    tool_args["slot"] = slot
                    tool_args["confirmed"] = True
                    if "property_name" not in tool_args:
                        tool_args["property_name"] = state.get_slot("preferred_project")

            # Followup / consultation request guard
            elif tool_name == "create_hr_followup":
                name = (
                    tool_args.get("name")
                    or tool_args.get("candidate_name")
                    or tool_args.get("caller_name")
                    or state.caller.name
                    or state.get_slot("name")
                    or state.get_slot("caller_name")
                )
                if not name:
                    execute_action = False
                    final_response_text = "I would be happy to have our team follow up with you. And may I have your name?"
                    decision.user_facing_response = final_response_text
                    state.set_pending_action("create_hr_followup")
                    state.set_pending_question("And may I have your name?")

            # Retired email tool path
            elif tool_name == "send_email":
                execute_action = False
                final_response_text = "I can share the requested property information through WhatsApp instead. Would you like me to send it that way?"
                decision.user_facing_response = final_response_text
                state.set_pending_action("send_message")
                state.set_pending_question(final_response_text)

            if execute_action:
                tool_call_req = ToolCallRequest(
                    tool_name=tool_name,
                    arguments=tool_args,
                    call_id=state.metadata.call_id,
                )
                t_tool = time.perf_counter()
                if tracker:
                    tracker.mark_tool_start()
                try:
                    tool_exec_result = await self.tool_provider.execute_tool(tool_call_req)
                except Exception as tool_err:
                    logger.exception("Tool execution raised for %s", tool_name)
                    tool_exec_result = ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        data={},
                        error_message=f"Tool execution failed: {type(tool_err).__name__}",
                    )
                timing["tool_execution_ms"] = round((time.perf_counter() - t_tool) * 1000, 2)
                if tracker:
                    tracker.mark_tool_end()

                if tool_exec_result.success:
                    # Verified Success: update state
                    state.complete_action(tool_name, tool_exec_result)
                    state.set_pending_question(None)

                    if tool_name == "update_business_status":
                        new_status = tool_exec_result.data.get("status") or tool_args.get("status")
                        if new_status:
                            state.update_business_status(str(new_status))

                    if tool_name == "send_message":
                        proj = tool_args.get("project_name") or state.get_slot("preferred_project") or "the requested property"
                        recipient = tool_args.get("recipient")
                        speech_confirmation = (
                            f"The WhatsApp message for {proj} was accepted by the messaging service for {recipient}. "
                            "I can't confirm recipient delivery unless the service reports it. Is there anything else I can help with?"
                        )
                    elif tool_name == "find_available_slots":
                        slots = tool_exec_result.data.get("available_slots") or tool_exec_result.data.get("slots") or []
                        proj = tool_args.get("property_name") or state.get_slot("preferred_project") or "that property"
                        date_str = tool_args.get("preferred_date") or "the requested date"
                        if slots:
                            state.update_slot("available_slots", [str(item) for item in slots])
                            slots_str = ", ".join(str(item) for item in slots[:3])
                            speech_confirmation = f"For {proj}, the available site-visit times on {date_str} are {slots_str}. Which one would you prefer?"
                            state.set_pending_action("create_calendar_event")
                            state.set_pending_question(f"Which available time would you prefer for {proj}?")
                        else:
                            state.update_slot("available_slots", [])
                            state.update_slot("visit_confirmation_pending", None)
                            state.update_slot("visit_confirmed", False)
                            speech_confirmation = f"I couldn't find an available site-visit time for {proj} on {date_str}. Would you like me to check another date?"
                            state.set_pending_action("find_available_slots")
                            state.set_pending_question("Would you like me to check another date?")
                    elif tool_name == "create_calendar_event":
                        slot = tool_args.get("slot") or tool_args.get("start_time") or "your requested time"
                        proj = tool_args.get("property_name") or state.get_slot("preferred_project") or "the property"
                        speech_confirmation = f"Your site visit to {proj} has been confirmed for {slot}. We look forward to welcoming you! Is there anything else I can assist you with?"
                    else:
                        speech_confirmation = "I have completed that request for you. Is there anything else I can assist you with?"

                    final_response_text = speech_confirmation.strip()
                    decision.user_facing_response = final_response_text
                else:
                    # Execution or verification failed
                    state.pending_action = None
                    state.last_tool_result = tool_exec_result
                    if tool_name == "send_message":
                        final_response_text = (
                            "I attempted to send the WhatsApp message, but encountered a system issue. "
                            "I have noted this down so our property team can share the details with you directly."
                        )
                    elif tool_name == "create_calendar_event":
                        final_response_text = (
                            "I attempted to confirm your site visit, but that slot could not be booked. "
                            "Would you like to check other available times?"
                        )
                    elif tool_name == "find_available_slots":
                        final_response_text = (
                            "I was unable to retrieve available site visit slots right now. "
                            "Would you like to try another date?"
                        )
                    else:
                        final_response_text = (
                            "I attempted to complete that request, but encountered a system issue. "
                            "I have noted this down so our team can follow up with you directly."
                        )
                    decision.user_facing_response = final_response_text
                    state.conversation_active = True

        # Guard: In outbound calls, NEVER allow agent to refer to itself by contact_name
        if state.metadata.direction == CallDirection.OUTBOUND:
            contact = (state.caller.name or state.contact_name or "").strip()
            if contact:
                for bad_phrase in [f"i am {contact.lower()}", f"i'm {contact.lower()}", f"my name is {contact.lower()}", f"this is {contact.lower()}"]:
                    if bad_phrase in final_response_text.lower():
                        if state.agent_name:
                            final_response_text = re.sub(re.escape(bad_phrase), f"I'm {state.agent_name}", final_response_text, flags=re.IGNORECASE)
                        else:
                            final_response_text = re.sub(re.escape(bad_phrase), "I'm calling", final_response_text, flags=re.IGNORECASE)
                        decision.user_facing_response = final_response_text

        # 12. Handle Clarification Questions
        if decision.needs_clarification and decision.clarification_question:
            state.set_pending_question(decision.clarification_question)
        elif not state.pending_question:
            state.set_pending_question(None)

        # 13. Record dialogue turns in transcript
        state.record_turn(
            role=TurnRole.CALLER,
            content=cleaned_msg,
            intent=decision.detected_intent,
        )
        state.record_turn(
            role=TurnRole.AGENT,
            content=final_response_text,
            citations=grounded_citations,
        )

        # Guard: Conversation remains active unless termination was requested
        conversation_active = state.conversation_active and not state.termination_requested

        total_turn_ms = round((time.perf_counter() - t_start) * 1000, 2)
        timing["total_turn_ms"] = total_turn_ms

        turn_idx = len(state.history) // 2
        logger.info(
            "\n"
            "======================================== [TURN DEBUG] ========================================\n"
            "TURN: %d\n"
            "INPUT: \"%s\"\n"
            "DIRECTION: %s\n"
            "DOMAIN: %s\n"
            "CURRENT INTENT: %s\n"
            "CURRENT STAGE: %s\n"
            "PROMPT SUMMARY: System [%d chars], User [%d chars]\n"
            "LLM PROVIDER: %s\n"
            "RAW GEMINI RESPONSE: %s\n"
            "PARSED DECISION: domain=%s, intent=%s, stage=%s, knowledge_req=%s, action_prop=%s\n"
            "VALIDATED DECISION: %s\n"
            "RAG REQUIRED: %s\n"
            "RAG QUERY: %s\n"
            "RAG RESULTS: %d chunks (%s)\n"
            "TOOL ACTION: %s\n"
            "FINAL RESPONSE: \"%s\"\n"
            "TOTAL LATENCY: %.2fms | Breakdown: %s\n"
            "==============================================================================================",
            turn_idx,
            cleaned_msg,
            state.metadata.direction.value,
            state.current_domain.value,
            decision.detected_intent,
            state.stage.value,
            len(system_instruction),
            len(user_prompt),
            self.llm.__class__.__name__,
            decision.user_facing_response if raw_llm_text is None else raw_llm_text,
            decision.detected_domain.value,
            decision.detected_intent,
            decision.proposed_stage.value,
            decision.knowledge_required,
            decision.action_proposed,
            decision.model_dump_json(exclude={"user_facing_response"}),
            decision.knowledge_required,
            decision.knowledge_query,
            len(grounded_citations),
            grounded_citations,
            decision.proposed_action.tool_name if (action_proposed and decision.proposed_action) else "None",
            final_response_text,
            total_turn_ms,
            timing,
        )

        if tracker:
            tracker.mark_response_ready()

        return EngineTurnResult(
            response_text=final_response_text,
            decision=decision,
            knowledge_required=decision.knowledge_required,
            knowledge_query=decision.knowledge_query,
            action_proposed=action_proposed,
            proposed_action=decision.proposed_action if action_proposed else None,
            conversation_active=conversation_active,
            termination_occurred=not conversation_active,
            grounded_citations=grounded_citations,
            tool_result=tool_exec_result,
            latency_ms=total_turn_ms,
            latency_breakdown=timing,
            raw_llm_response=raw_llm_text or decision.user_facing_response,
        )
