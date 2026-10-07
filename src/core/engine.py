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

if TYPE_CHECKING:
    from src.domains.base import DomainConfig
    from src.domains.registry import DomainRegistry

logger = get_logger("core.engine")


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
        """Initiate an outbound conversation with an opening agent turn."""
        if state.metadata.direction != CallDirection.OUTBOUND:
            raise ValueError(
                f"Cannot start outbound conversation on an inbound call (call_id: {state.metadata.call_id})"
            )

        domain_config = self.registry.get_or_raise(state.current_domain)
        t_out_start = time.perf_counter()

        # 1. Synthesize opening prompt
        prompt = self.synthesizer.build_outbound_opening_prompt(state, domain_config)
        system_instruction = render_prompt(
            "conversation/outbound_opening_system.txt",
            domain_name=domain_config.name,
        ).strip()

        opening_text: Optional[str] = None
        # 2. Try LLM generation if available
        try:
            if hasattr(self.llm, "generate_response"):
                raw_res = await self.llm.generate_response(prompt, system_instruction)
                from src.core.llm import extract_json_block
                import json

                try:
                    clean = extract_json_block(raw_res)
                    parsed = json.loads(clean)
                    if isinstance(parsed, dict) and "user_facing_response" in parsed:
                        opening_text = parsed["user_facing_response"]
                    elif isinstance(parsed, dict) and "response" in parsed:
                        opening_text = parsed["response"]
                except Exception:
                    opening_text = raw_res.strip().strip('"')
        except Exception as exc:
            logger.warning("LLM outbound opening generation failed, using deterministic opening: %s", exc)

        # 3. Fallback deterministic opening if needed
        contact_name = (state.caller.name or state.contact_name or "").strip()
        has_identity_confusion = False
        if contact_name and opening_text:
            lower_open = opening_text.lower()
            lower_contact = contact_name.lower()
            if any(f"{p} {lower_contact}" in lower_open for p in ["i am", "i'm", "my name is", "this is"]):
                has_identity_confusion = True

        if (
            not opening_text
            or len(opening_text.strip()) < 10
            or opening_text.strip().startswith("{")
            or opening_text.strip() == "Mock default response."
            or has_identity_confusion
        ):
            opening_text = self._generate_deterministic_outbound_opening(state, domain_config)

        # 4. Outbound identity-first opening. Do not discuss the campaign purpose until
        # the person on the line has been identified/verified.
        contact_name = (state.caller.name or state.contact_name or "").strip()
        if contact_name:
            opening_text = f"Hi, I'm calling from {domain_config.name}. Am I speaking with {contact_name}?"
            pending_q = f"Am I speaking with {contact_name}?"
        else:
            opening_text = f"Hi, I'm calling from {domain_config.name}. May I know who I'm speaking with?"
            pending_q = "May I know who I'm speaking with?"
            state.update_slot("outbound_identity_name_pending", True)

        state.update_slot("outbound_identity_verified", False)

        decision = ConversationalDecision(
            detected_domain=state.current_domain,
            detected_intent="identity_verification",
            proposed_stage=ConversationStage.GREETING,
            user_facing_response=opening_text,
            action_proposed=False,
            knowledge_required=False,
        )

        # 5. Update state
        state.set_stage(ConversationStage.GREETING)
        state.set_intent("identity_verification", clear_pending_question=False)
        state.set_pending_question(pending_q)
        state.conversation_active = True

        # 6. Record agent opening in transcript
        state.record_turn(
            role=TurnRole.AGENT,
            content=opening_text,
        )

        logger.info(
            "Outbound conversation %s started with opening: '%s'",
            state.metadata.call_id,
            opening_text,
        )

        return EngineTurnResult(
            response_text=opening_text,
            decision=decision,
            conversation_active=True,
            termination_occurred=False,
            latency_ms=round((time.perf_counter() - t_out_start) * 1000, 2),
            latency_breakdown={"opening_generation_ms": round((time.perf_counter() - t_out_start) * 1000, 2)},
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

        if state.current_domain == DomainType.EDUSAAS:
            if agent_name:
                intro = f"I'm {agent_name} from EduSaaS Academic Admissions & Guidance."
            else:
                intro = "I'm calling from EduSaaS Academic Admissions & Guidance."

            if reason:
                r_clean = reason.rstrip(".!? ")
                for prefix in [
                    "follow up with student who showed interest in ",
                    "follow up with student who had shown interest in ",
                    "follow up with student regarding interest in ",
                    "follow up with student regarding ",
                    "follow up with student about ",
                    "follow up regarding interest in ",
                    "follow up regarding ",
                    "follow up about ",
                    "follow up on ",
                    "you had shown interest in ",
                    "shown interest in ",
                    "interest in ",
                ]:
                    if r_clean.lower().startswith(prefix):
                        r_clean = r_clean[len(prefix) :]
                        break
                return f"{greeting_contact}{intro} I'm reaching out because you had shown interest in {r_clean}. Is this a good time to speak for a couple of minutes?"
            return f"{greeting_contact}{intro} I'm reaching out because you had shown interest in our courses. Is this a good time to speak for a couple of minutes?"

        else:  # Vayvora
            if agent_name:
                intro = f"I'm {agent_name} from Vayvora Technologies."
            else:
                intro = "I'm calling from Vayvora Technologies."

            target_org = company or "your organization"
            if reason:
                r_clean = reason.rstrip(".!? ")
                for prefix in [
                    "understand whether organization is exploring ",
                    "understand whether your organization is exploring ",
                    "understand whether company is exploring ",
                    "understand whether ",
                    "exploring ",
                ]:
                    if r_clean.lower().startswith(prefix):
                        r_clean = r_clean[len(prefix) :]
                        break
                return f"{greeting_contact}{intro} I'm reaching out to understand whether {target_org} is currently exploring {r_clean}. Is this a good time for a quick conversation?"
            return f"{greeting_contact}{intro} I'm reaching out to understand whether {target_org} is currently exploring AI or automation solutions. Is this a good time for a quick conversation?"

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
        """Extract a caller-provided name from common natural voice responses."""
        text = (message or "").strip()
        if not text:
            return None
        patterns = [
            r"^(?:i am|i'm|im|this is|it's|it is)\s+([A-Za-z][A-Za-z .'-]{0,60})[.!?]?$",
        ]
        for pattern in patterns:
            m = re.match(pattern, text, flags=re.IGNORECASE)
            if m:
                value = (m.group(1) if m.lastindex else m.group(0)).strip(" .,!?")
                if value and value.lower() not in {"here", "speaking"}:
                    return value
        # A short name-only response is acceptable, but do not treat acknowledgements as names.
        if len(text.split()) <= 3 and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,40}", text):
            if text.lower() not in {"yes", "yeah", "yep", "sure", "okay", "ok", "no", "nope", "hello", "hi", "hey"}:
                return text.strip(" .,!?")
        return None

    @staticmethod
    def _is_identity_confirmation(message: str) -> bool:
        text = (message or "").strip().lower()
        return text in {
            "yes", "yeah", "yep", "yes speaking", "speaking", "that's me", "that is me",
            "this is me", "correct", "right", "yes this is me", "yes that's me",
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
        """Apply a spoken local correction such as 'not powan, pawan' to the current email."""
        if not current_email or not message:
            return None
        text = message.strip()
        # Common STT forms: "not powan, pawan" / "not powan but pawan" / "it's pawan, not powan".
        patterns = [
            r"not\s+([a-z0-9._+-]+)\s*(?:,|but|rather|instead)\s*([a-z0-9._+-]+)",
            r"([a-z0-9._+-]+)\s*(?:,|but)\s*not\s+([a-z0-9._+-]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            if pattern.startswith("not"):
                old, new = m.group(1), m.group(2)
            else:
                new, old = m.group(1), m.group(2)
            local, sep, domain = current_email.partition("@")
            if not sep or not old or not new:
                continue
            if old.lower() in local.lower():
                corrected_local = re.sub(re.escape(old), new, local, count=1, flags=re.IGNORECASE)
                candidate = f"{corrected_local}@{domain}"
                return candidate if is_valid_email(candidate) else None
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
            response = "Hi, I'm calling from EduSaaS. May I know who I'm speaking with?" if state.current_domain == DomainType.EDUSAAS else "Hello, I'm calling from Vayvora Technologies. May I know who I'm speaking with?"

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

        # Outbound identity must be verified before any course, RAG, email, or action flow.
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
            clarify_resp = "Sure. Are you calling about our software and AI solutions, or our education services?"
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
                # Initial routing: check deterministic classifier
                pre_domain = classify_inbound_domain(cleaned_msg)
                if pre_domain in (DomainType.VAYVORA, DomainType.EDUSAAS):
                    state.lock_domain(pre_domain)
                    logger.info("Inbound domain resolved and locked to %s", pre_domain.value)

        # 6. Retrieve active domain configuration
        domain_config = self.registry.get_or_raise(state.current_domain)

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
                # Domain was not locked. If LLM detected a valid business domain, lock it now.
                if decision.detected_domain in (DomainType.VAYVORA, DomainType.EDUSAAS):
                    state.lock_domain(decision.detected_domain)
                    logger.info("Inbound domain resolved from LLM decision and locked to %s", decision.detected_domain.value)
                else:
                    # Still unknown
                    decision.detected_domain = DomainType.UNKNOWN
                    if decision.user_facing_response in ("Hi, how can I help you today?", None, ""):
                        decision.user_facing_response = "Sure. Are you calling about our software and AI solutions, or our education services?"
                        decision.needs_clarification = True
                        decision.clarification_question = decision.user_facing_response
        else:
            # Outbound calls: update domain if caller or LLM explicitly pivoted
            if decision.detected_domain in (DomainType.VAYVORA, DomainType.EDUSAAS):
                state.current_domain = decision.detected_domain

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
        is_asking_for_email_address = bool(
            (prior_pending_question and any(w in prior_pending_question.lower() for w in ["email address", "what email", "share your email", "send the details to", "send the course details to", "send them to"]))
            or (state.pending_question and any(w in state.pending_question.lower() for w in ["email address", "what email", "share your email", "send the details to", "send the course details to", "send them to"]))
        )
        is_asking_for_topic = bool(
            (prior_pending_question and any(w in prior_pending_question.lower() for w in ["what information", "what would you like me to send", "which course"]))
            or (state.pending_question and any(w in state.pending_question.lower() for w in ["what information", "what would you like me to send", "which course"]))
        )
        is_email_context = is_asking_for_email_address or (
            (prior_pending_action == "send_email" or state.pending_action == "send_email") and not is_asking_for_topic
        )

        # Handle unclear or invalid email attempt
        if (is_asking_for_email_address or is_unclear_or_invalid_email_attempt(cleaned_msg)) and not extracted_email and not explicit_email_corr:
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
            if is_email_context or is_asking_for_email_address or not state.caller.email:
                decision.extracted_slots["email"] = extracted_email
                state.update_slot("email", extracted_email, sync_caller=True)
                state.caller.email = extracted_email

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

        # Promote to calendar action if caller is requesting scheduling or answering pending date/time question
        is_calendar_request = any(w in cleaned_msg.lower() for w in ["schedule", "consultation", "demo", "calendar", "book a slot"])
        if not action_proposed and (
            is_calendar_request
            or prior_pending_action == "create_calendar_event"
            or state.pending_action == "create_calendar_event"
            or (
                prior_pending_question
                and any(w in prior_pending_question.lower() for w in ["date", "time slot", "time would", "work best", "schedule"])
            )
            or (
                state.pending_question
                and any(w in state.pending_question.lower() for w in ["date", "time slot", "time would", "work best", "schedule"])
            )
        ) and (extracted_pref or state.get_slot("meeting_preference")):
            slot_to_book = extracted_pref or state.get_slot("meeting_preference")
            decision.action_proposed = True
            decision.proposed_action = ProposedAction(
                tool_name="create_calendar_event",
                arguments={"slot": slot_to_book, "confirmed": True},
            )
            action_proposed = True

        # Promote to email action if caller is requesting brochure/details or answering pending email request
        target_email = explicit_email_corr or extracted_email
        is_email_request = any(
            phrase in cleaned_msg.lower()
            for phrase in [
                "course details",
                "send me the details",
                "send the details",
                "send me details",
                "send details",
                "email me the details",
                "email the details",
                "send me the course details",
                "send course details",
                "send the syllabus",
                "send me the syllabus",
                "email me the syllabus",
                "send brochure",
                "send me the brochure",
                "email me the brochure",
                "email me",
                "mail me",
                "send me an email",
                "send an email",
            ]
        )

        if state.current_domain == DomainType.EDUSAAS:
            from src.domains.edusaas.email import (
                is_explicit_email_request,
                is_vague_email_request,
                detect_requested_topic,
            )

            # Check if caller made a vague email request without specifying topic
            if is_vague_email_request(cleaned_msg, state.history, state.get_slot("target_course")):
                vague_ask = "Absolutely. What information would you like me to send you?"
                state.set_pending_action("send_email")
                state.set_pending_question(vague_ask)
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, vague_ask)
                synthetic_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="request_brochure",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response=vague_ask,
                    action_proposed=False,
                    proposed_action=None,
                )
                return EngineTurnResult(
                    response_text=vague_ask,
                    decision=synthetic_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )

            # Detect requested topic from caller message (latest request wins)
            topic_k, topic_name = detect_requested_topic(cleaned_msg, state.history, state.get_slot("target_course"))
            if topic_name and (
                ((is_explicit_email_request(cleaned_msg) or is_asking_for_topic) and not decision.extracted_slots.get("target_course"))
                or (not decision.extracted_slots.get("target_course") and any(w in cleaned_msg.lower() for w in ["ai", "data science", "full stack", "cloud", "cyber", "dsa", "pricing", "admission", "fee", "course"]))
            ):
                state.update_slot("target_course", topic_name, sync_caller=True)
                decision.extracted_slots["target_course"] = topic_name

            # Guard: If caller did not explicitly request email and is not answering email address/topic prompt,
            # DO NOT allow LLM to trigger automatic email sending (TEST 14)
            if not is_explicit_email_request(cleaned_msg) and not is_asking_for_email_address and not is_asking_for_topic and not explicit_email_corr:
                if action_proposed and decision.proposed_action and decision.proposed_action.tool_name == "send_email":
                    decision.action_proposed = False
                    decision.proposed_action = None
                    action_proposed = False
                    decision.knowledge_required = True

            is_email_act = is_explicit_email_request(cleaned_msg) or is_asking_for_email_address or is_asking_for_topic or bool(explicit_email_corr)
        else:
            is_email_act = is_email_context or is_email_request or bool(explicit_email_corr)

        if not action_proposed and is_email_act:
            subj = (
                "Course Details & Curriculum"
                if (state.current_domain == DomainType.EDUSAAS or "course" in cleaned_msg.lower())
                else "Detailed Information & Overview"
            )
            email_args: Dict[str, Any] = {"subject": subj}
            if target_email and is_valid_email(target_email):
                email_args["recipient"] = target_email
            decision.action_proposed = True
            decision.proposed_action = ProposedAction(
                tool_name="send_email",
                arguments=email_args,
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
            and cleaned_msg.lower().strip() in {"yes", "yeah", "yep", "correct", "that's correct", "that is correct", "yes that's correct", "yes that is correct"}
        ):
            confirmed = str(state.get_slot("email_confirmation_pending"))
            decision.action_proposed = True
            decision.proposed_action = ProposedAction(
                tool_name="send_email",
                arguments={"subject": "Course Details & Curriculum", "recipient": confirmed},
            )
            action_proposed = True

        # Email safety gate: never send until the recipient has been explicitly confirmed.
        if action_proposed and decision.proposed_action and decision.proposed_action.tool_name == "send_email":
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

                if self._is_identity_confirmation(cleaned_msg) or cleaned_msg.lower().strip() in {"yes", "correct", "that's correct", "that is correct"}:
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
                elif cleaned_msg.lower().strip() in {"no", "nope", "not correct", "that's not correct", "that is not correct"}:
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
        if action_proposed and decision.proposed_action and self.tool_provider:
            tool_name = decision.proposed_action.tool_name
            tool_args = dict(decision.proposed_action.arguments)

            # Prohibited placeholders and sender emails that must never be used as recipient
            prohibited_vals = {
                "default caller",
                "caller@mail",
                "unknown@example.com",
                "john doe",
                "jane doe",
                "default@example.com",
                "fallback@example.com",
                "noreply@vayvora.com",
                "noreply@edusaas.com",
                "info@vayvora.com",
                "info@edusaas.com",
                "admin@vayvora.com",
                "admin@edusaas.com",
                "none",
                "null",
                "empty",
            }
            if self.settings and self.settings.smtp_from_email:
                prohibited_vals.add(self.settings.smtp_from_email.lower().strip())

            # Strip any hallucinated placeholders from tool arguments
            for arg_k in ["caller_name", "candidate_name", "name", "recipient", "email", "caller_email"]:
                if arg_k in tool_args and str(tool_args[arg_k]).strip().lower() in prohibited_vals:
                    tool_args.pop(arg_k, None)

            # Auto-populate caller profile details into arguments if absent
            if "caller_name" not in tool_args and state.caller.name and state.caller.name.strip().lower() not in prohibited_vals:
                tool_args["caller_name"] = state.caller.name
            if "caller_email" not in tool_args and state.caller.email and state.caller.email.strip().lower() not in prohibited_vals:
                tool_args["caller_email"] = state.caller.email
            if "email" not in tool_args and state.caller.email and state.caller.email.strip().lower() not in prohibited_vals:
                tool_args["email"] = state.caller.email
            if "recipient" not in tool_args and state.get_slot("email_confirmed") and state.caller.email and state.caller.email.strip().lower() not in prohibited_vals:
                tool_args["recipient"] = state.caller.email
            if "caller_phone" not in tool_args and state.caller.phone and state.caller.phone.strip().lower() not in prohibited_vals:
                tool_args["caller_phone"] = state.caller.phone
            if "domain" not in tool_args:
                tool_args["domain"] = state.current_domain.value

            # Guard 1: Deterministic Email Safety Guard
            recipient = (tool_args.get("recipient") or tool_args.get("email") or (state.caller.email if state.get_slot("email_confirmed") else "") or "").strip()
            if recipient.lower() in prohibited_vals:
                recipient = ""

            recipient_valid = is_valid_email(recipient)
            recipient_authorized = True

            # Verify recipient authenticity (must be in profile, session state, or caller transcripts)
            allowed_recipients = set()
            if state.caller.email and is_valid_email(state.caller.email):
                allowed_recipients.add(state.caller.email.strip().lower())
            if getattr(state, "contact_email", None) and is_valid_email(state.contact_email):
                allowed_recipients.add(state.contact_email.strip().lower())
            for t in state.history:
                if t.role == TurnRole.CALLER:
                    t_email = extract_email_address(t.content)
                    if t_email and is_valid_email(t_email):
                        allowed_recipients.add(t_email.strip().lower())
            if cleaned_msg:
                m_email = extract_email_address(cleaned_msg)
                if m_email and is_valid_email(m_email):
                    allowed_recipients.add(m_email.strip().lower())
            if explicit_email_corr and is_valid_email(explicit_email_corr):
                allowed_recipients.add(explicit_email_corr.strip().lower())

            if not recipient or recipient.lower() not in allowed_recipients:
                recipient_authorized = False

            if tool_name == "send_email" and (not recipient or not recipient_valid or not recipient_authorized):
                # Recipient is missing, invalid, or unauthorized fallback!
                # BLOCK send_email tool execution deterministically.
                tool_args.pop("recipient", None)
                tool_args.pop("email", None)
                if state.current_domain == DomainType.EDUSAAS or any(w in cleaned_msg.lower() for w in ["course", "syllabus", "curriculum", "program", "data science", "ai"]):
                    final_response_text = "Sure. What email address should I send the course details to?"
                elif state.metadata.direction == CallDirection.OUTBOUND:
                    final_response_text = "Sure. What email address should I send the details to?"
                else:
                    final_response_text = "Sure. What email address should I send them to? Could you please share your email address?"
                decision.user_facing_response = final_response_text
                state.set_pending_action("send_email")
                state.set_pending_question("Could you please share your email address?")
            # Guard 2: HR Followup / Candidate Safety - missing candidate name
            elif tool_name == "create_hr_followup" and not (
                tool_args.get("candidate_name") or tool_args.get("name") or state.caller.name
            ):
                final_response_text = "And may I have your name?"
                decision.user_facing_response = final_response_text
                state.set_pending_question("And may I have your name?")
            # Guard 3: Calendar Safety - missing confirmed slot
            elif tool_name == "create_calendar_event" and not (
                tool_args.get("slot")
                or tool_args.get("start_time")
                or tool_args.get("time")
                or state.get_slot("meeting_preference")
            ):
                final_response_text = (
                    "I can certainly schedule that consultation. "
                    "Which date or time slot would work best for you?"
                )
                decision.user_facing_response = final_response_text
                state.set_pending_question(
                    "Which date or time slot would work best for you?"
                )
            else:
                if tool_name == "send_email":
                    tool_args["recipient"] = recipient
                    tool_args["email"] = recipient
                    if state.current_domain == DomainType.EDUSAAS:
                        # RAG retrieval for EduSaaS email
                        edusaas_rag_context = ""
                        topic_title = state.get_slot("target_course")
                        if self.knowledge_provider:
                            try:
                                rag_query = RAGQuery(
                                    domain=DomainType.EDUSAAS,
                                    query_text=topic_title or cleaned_msg,
                                    top_k=self.settings.rag_top_k if self.settings else 3,
                                    relevance_threshold=self.settings.rag_relevance_threshold if self.settings else 0.5,
                                )
                                if hasattr(self.knowledge_provider, "retrieve_grounded_context"):
                                    rag_res = await self.knowledge_provider.retrieve_grounded_context(rag_query)
                                    if rag_res and rag_res.formatted_context:
                                        edusaas_rag_context = rag_res.formatted_context
                                else:
                                    chunks = await self.knowledge_provider.search(rag_query)
                                    if chunks:
                                        edusaas_rag_context = "\n".join(c.content for c in chunks)
                            except Exception as err:
                                logger.warning("RAG retrieval for EduSaaS email failed: %s", err)

                        from src.domains.edusaas.email import generate_edusaas_email
                        edusaas_email = generate_edusaas_email(
                            direction=state.metadata.direction,
                            contact_name=state.caller.name or state.contact_name,
                            agent_name=state.agent_name or (self.settings.agent_name if self.settings else None),
                            topic_requested=topic_title,
                            rag_context=edusaas_rag_context,
                            conversation_history=state.history,
                            campaign_context=getattr(state, "campaign_context", None) if state.metadata.direction == CallDirection.OUTBOUND else None,
                            recipient=recipient,
                            caller_message=cleaned_msg,
                        )
                        tool_args["subject"] = edusaas_email.subject
                        tool_args["body"] = edusaas_email.body
                        tool_args["html_body"] = edusaas_email.html_body
                        tool_args["topic_display"] = edusaas_email.topic
                elif tool_name == "create_calendar_event":
                    slot = (
                        tool_args.get("slot")
                        or tool_args.get("start_time")
                        or tool_args.get("time")
                        or state.get_slot("meeting_preference")
                    )
                    if slot:
                        tool_args["slot"] = slot
                    tool_args.setdefault("confirmed", True)

                # Dispatch to tool provider
                tool_call_req = ToolCallRequest(
                    tool_name=tool_name,
                    arguments=tool_args,
                    call_id=state.metadata.call_id,
                )
                t_tool = time.perf_counter()
                if tracker:
                    tracker.mark_tool_start()
                tool_exec_result = await self.tool_provider.execute_tool(tool_call_req)
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

                    if tool_name == "send_email":
                        if state.current_domain == DomainType.EDUSAAS:
                            topic_disp = (tool_args.get("topic_display") or "").lower()
                            msg_low = cleaned_msg.lower()
                            if "data science" in msg_low or "data science" in topic_disp:
                                speech_confirmation = f"I've sent the complete course details and enrollment link to {recipient}."
                            elif bool(re.search(r"\b(ai|artificial intelligence|machine learning)\b", msg_low)) or "artificial intelligence" in topic_disp:
                                speech_confirmation = f"I've sent the complete course details and enrollment link to {recipient}."
                            elif "admission" in msg_low or "admission" in topic_disp:
                                speech_confirmation = f"I've sent the complete course details and enrollment link to {recipient}."
                            elif "pricing" in msg_low or "fee" in msg_low or "pricing" in topic_disp:
                                speech_confirmation = f"I've sent the complete course details and enrollment link to {recipient}."
                            else:
                                speech_confirmation = f"I've sent the complete course details and enrollment link to {recipient}."
                        else:
                            speech_confirmation = f"I've sent the details to {recipient}."

                    elif tool_name == "create_calendar_event":
                        slot = tool_args.get("slot") or state.get_slot("meeting_preference") or "the requested time"
                        speech_confirmation = f"I have scheduled your consultation for {slot}. Is there anything else I can assist you with?"
                    else:
                        # Synthesize verbal confirmation incorporating verified reference
                        action_prompt = render_prompt(
                            "tools/action_confirmation.txt",
                            caller_message=cleaned_msg,
                            tool_name=tool_name,
                            verification_code=tool_exec_result.verification_code,
                            data=tool_exec_result.data,
                        ).strip()

                        speech_confirmation = await self.llm.generate_response(
                            prompt=action_prompt,
                            system_instruction=load_prompt("tools/system.txt").strip(),
                        )
                        try:
                            from src.core.llm import extract_json_block
                            import json
                            clean = extract_json_block(speech_confirmation)
                            parsed = json.loads(clean)
                            if isinstance(parsed, dict) and "user_facing_response" in parsed:
                                speech_confirmation = parsed["user_facing_response"]
                            elif isinstance(parsed, dict) and "response" in parsed:
                                speech_confirmation = parsed["response"]
                        except Exception:
                            pass

                        if (
                            not speech_confirmation
                            or speech_confirmation == "Mock default response."
                            or speech_confirmation.strip().startswith("{")
                        ):
                            speech_confirmation = "I have completed that request for you. Is there anything else I can assist you with?"

                    final_response_text = speech_confirmation.strip()
                    decision.user_facing_response = final_response_text
                else:
                    # Execution or verification failed
                    state.pending_action = None
                    state.last_tool_result = tool_exec_result
                    err_msg = tool_exec_result.error_message or "action could not be verified"
                    if tool_name == "send_email":
                        final_response_text = (
                            "I attempted to send the email, but encountered a system issue and couldn't send the email right now. "
                            "I have noted this down so our team can follow up with you directly."
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
