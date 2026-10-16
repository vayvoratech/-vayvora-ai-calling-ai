from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.core.errors import StreamingCallbackError
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
from src.rag.retrieval.catalog_intent import is_catalog_query
from src.state.models import ConversationState

from src.tools.message_provider import validate_phone_number
from src.database.repository import mask_email, mask_phone_number, mask_sensitive_text

if TYPE_CHECKING:
    from src.domains.base import DomainConfig
    from src.domains.registry import DomainRegistry

logger = get_logger("core.engine")


def _extract_site_visit_date_time(text: str) -> tuple[Optional[str], Optional[str]]:
    """Split a caller's date/time phrase into separate lexical preferences.

    This deliberately does not resolve relative dates; the calendar layer remains
    responsible for interpreting "tomorrow" in its configured timezone.
    """
    if not isinstance(text, str) or not text.strip():
        return None, None

    date_pattern = re.compile(
        r"\b(today|tomorrow|yesterday|(?:(?:this|next)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?)\b",
        re.IGNORECASE,
    )
    time_pattern = re.compile(
        r"\b(\d{1,2}(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)|noon|midnight)\b",
        re.IGNORECASE,
    )
    date_match = date_pattern.search(text)
    time_match = time_pattern.search(text)
    date_value = re.sub(r"\s+", " ", date_match.group(1)).strip() if date_match else None
    time_value = re.sub(r"\s+", " ", time_match.group(1)).strip() if time_match else None
    return date_value, time_value


def _time_minutes(value: Any) -> Optional[int]:
    """Normalize a time phrase such as '10 am' or '10:00 AM' to minutes after midnight."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower().replace('.', '')
    if cleaned == 'noon':
        return 12 * 60
    if cleaned == 'midnight':
        return 0
    match = re.search(r"\b(\d{1,2})(?::([0-5]\d))?\s*(am|pm)\b", cleaned)
    if not match:
        return None
    hour = int(match.group(1)) % 12
    minute = int(match.group(2) or 0)
    if match.group(3) == 'pm':
        hour += 12
    return hour * 60 + minute


def _slot_matches_requested_time(slot: Any, requested_time: Any) -> bool:
    wanted = _time_minutes(str(requested_time or ''))
    if wanted is None:
        return False
    matches = re.findall(
        r"\b(\d{1,2})(?::([0-5]\d))?\s*(a\.?m\.?|p\.?m\.?)\b",
        str(slot or ''),
        re.IGNORECASE,
    )
    for hour_text, minute_text, meridiem in matches:
        hour = int(hour_text) % 12
        if meridiem.lower().startswith('p'):
            hour += 12
        if hour * 60 + int(minute_text or 0) == wanted:
            return True
    return False


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
    """Determine whether decision.user_facing_response can be safely reused.

    Pre-retrieval draft responses cannot safely bypass grounded RAG synthesis
    when knowledge is required for factual queries, because the initial draft
    was generated before verified knowledge context was retrieved.
    Always returns False to guarantee grounded synthesis.
    """
    return False


_INFRA_PROJECT_ALIASES = {
    "jb pride": ("jb pride",),
    "nature valley": ("nature valley", "jb's nature valley", "jbs nature valley"),
    "pristine city": ("pristine city",),
    "serene city": ("serene city", "jb's serene city", "jbs serene city"),
}


def _is_infra_pricing_query(message: str) -> bool:
    """Detect property-price/fee questions that must always be grounded in RAG."""
    text = (message or "").lower()
    cues = (
        "price", "pricing", "cost", "rate", "per square yard", "per sq yard",
        "per sq. yard", "per sq yd", "square-yard", "square yard rate",
        "booking amount", "booking fee", "down payment", "registration fee",
        "registration charges", "development fee", "development charges",
        "additional charges", "hidden charges", "total cost", "plot cost",
        "rupees", "₹", "inr", " rs ", "rs.",
    )
    return any(cue in text for cue in cues) or bool(re.search(
        r"\bhow much\b.{0,60}\b(?:cost|price|pay|rupees|rate|charge)\b",
        text,
    ))


def _project_for_pricing_query(message: str, preferred_project: Optional[str] = None) -> Optional[str]:
    """Resolve only a project explicitly named by the caller or stored in trusted state."""
    text = (message or "").lower()
    for canonical, aliases in _INFRA_PROJECT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    preferred = (preferred_project or "").lower().strip()
    for canonical, aliases in _INFRA_PROJECT_ALIASES.items():
        if preferred == canonical or any(alias in preferred for alias in aliases):
            return canonical
    return None


def _chunk_supports_project_price(chunk: Any, project_name: str, query: str) -> bool:
    """Require a price amount and its matching unit/fee label on the same source line."""
    content = str(getattr(chunk, "content", "") or "")
    title = str(getattr(chunk, "title", "") or "")
    doc_id = str(getattr(chunk, "doc_id", "") or "")
    source_text = "\n".join((title, doc_id, content)).lower()
    aliases = _INFRA_PROJECT_ALIASES.get(project_name, (project_name,))
    if not any(alias in source_text for alias in aliases):
        return False

    query_lower = (query or "").lower()
    asks_per_area_rate = any(cue in query_lower for cue in (
        "per square yard", "per sq yard", "per sq. yard", "per sq yd",
        "square yard rate", "per-square-yard",
    ))
    amount_pattern = re.compile(
        r"(?:₹|\binr\b|\brs\.?\s*)\s*[0-9][0-9,]*(?:\.\d+)?|"
        r"\b[0-9][0-9,]*(?:\.\d+)?\s*(?:rupees|rs\.?|inr)\b",
        flags=re.IGNORECASE,
    )
    unit_pattern = re.compile(
        r"(?:per\s+(?:square|sq\.?)[-\s]*yard|per\s+sq\.?\s*yd|"
        r"(?:square|sq\.?)[-\s]*yard\s*(?:rate|price))",
        flags=re.IGNORECASE,
    )
    price_label_pattern = re.compile(
        r"\b(price|pricing|cost|rate|booking|registration|development|charges?|fee|fees|deposit|down payment|per\s+(?:square|sq\.?)[-\s]*yard)\b",
        flags=re.IGNORECASE,
    )
    for line in content.splitlines():
        if not amount_pattern.search(line):
            continue
        if asks_per_area_rate and unit_pattern.search(line):
            return True
        if not asks_per_area_rate and price_label_pattern.search(line):
            return True
    return False


def _has_verified_price_evidence(chunks: List[Any], project_name: Optional[str], query: str) -> bool:
    """Return true only when a retrieved chunk explicitly supports the requested project's price."""
    if not project_name:
        return False
    return any(_chunk_supports_project_price(chunk, project_name, query) for chunk in (chunks or []))


def _is_infra_pricing_meta_query(message: str) -> bool:
    """Detect questions about whether/how pricing is documented, not a request for the price itself."""
    text = re.sub(r"\s+", " ", (message or "").lower()).strip()
    asks_about_records = any(term in text for term in (
        "knowledge base", "knowledge-base", "stored", "in the source", "in your source",
        "source document", "source documents", "retrieved", "listed", "documented",
        "what information", "what details", "what do you have", "does it say",
        "is it mentioned", "is it listed", "do you have a record", "show me the source",
    ))
    asks_about_pricing = any(term in text for term in (
        "price", "pricing", "cost", "rate", "per square yard", "per sq yard",
        "booking amount", "registration fee", "charges", "amount", "rupees", "₹", "inr",
    ))
    return asks_about_records and asks_about_pricing


def _is_explicit_project_followup_request(message: str) -> bool:
    """Recognize an explicit request to record a project-team follow-up, excluding refusals."""
    text = re.sub(r"\s+", " ", (message or "").lower()).strip()
    if not text:
        return False
    if re.search(
        r"\b(?:don't|dont|do not|never|no|not|cancel|stop)\b.{0,45}\b"
        r"(?:follow.?up|call|contact|arrange|ask the team|project team)\b",
        text,
    ):
        return False
    return bool(re.search(
        r"(?:arrange|request|schedule|set up|have|ask|tell|get)\b.{0,70}\b"
        r"(?:follow.?up|callback|call me|contact me|project team|team to confirm|team to call)\b"
        r"|\b(?:please )?(?:have|ask|get) (?:the )?(?:project )?team\b.{0,70}\b"
        r"(?:confirm|call|contact|follow.?up)\b"
        r"|\b(?:call me back|have someone call me|please contact me)\b",
        text,
    ))


def _project_from_conversation_context(
    history: Any,
    current_message: str,
    preferred_project: Optional[str] = None,
) -> Optional[str]:
    """Resolve the project from trusted state, the current utterance, or recent caller turns."""
    project = _project_for_pricing_query(current_message, preferred_project=preferred_project)
    if project:
        return project
    for turn in reversed(history or []):
        role = str(getattr(turn, "role", "")).lower()
        if "caller" not in role:
            continue
        project = _project_for_pricing_query(str(getattr(turn, "content", "") or ""))
        if project:
            return project
    return None


def _is_explicit_action_confirmation(message: str) -> bool:
    """Accept clear positive confirmations, never confirmations embedded in a refusal."""
    text = re.sub(r"\s+", " ", (message or "").lower()).strip().strip(" .!?")
    if not text:
        return False
    no_problem_prefix = text.startswith(("no problem", "no worries"))
    if (
        (text.startswith(("no ", "no,", "nope", "not now", "not at all")) and not no_problem_prefix)
        or re.search(
            r"\b(?:don't|dont|do not|never|not|cancel|stop|can't|cannot|won't|wont)\b.{0,35}\b"
            r"(?:book|confirm|schedule|reserve|site visit|slot|booking)\b",
            text,
        )
    ):
        return False
    exact = {
        "yes", "yeah", "yep", "sure", "confirm", "confirmed", "please book",
        "book it", "that works", "sounds good", "yes please", "correct", "perfect",
        "go ahead", "please do", "that time works", "yes book it", "no problem, book it",
        "no problem book it", "no worries, book it",
    }
    return text in exact or bool(re.search(
        r"\b(?:please book it|please book the visit|go ahead and book|yes,? book|"
        r"yes,? (?:that works|sounds good|please do|confirm)|i confirm(?: the)? (?:booking|slot|visit)|"
        r"confirm the booking|that works(?: for me)?|works for me|that time works for me|"
        r"please confirm the booking|go ahead and schedule)\b",
        text,
    ))


def _is_appointment_cancellation_request(message: str) -> bool:
    """Detect an explicit request to cancel or delete an existing appointment or booking."""
    text = re.sub(r"\s+", " ", (message or "").lower()).strip()
    if not text:
        return False
    # Exclude negative cancellations ("don't cancel", "do not cancel", "never cancel")
    if re.search(r"\b(?:don't|dont|do not|never|stop|not)\b.{0,25}\b(?:cancel|delete|drop|remove)\b", text):
        return False
    # Exclude refusal to create new booking ("don't want to book", "do not book", "no need to book")
    if re.search(r"\b(?:don't|dont|do not|never|no need to|not)\b.{0,35}\b(?:book|booking|schedule|reserve|site visit)\b", text):
        return False
    return bool(re.search(
        r"\b(?:cancel|delete|drop|remove)\b.{0,35}\b(?:appointment|booking|site visit|visit|tour|slot|reservation|meeting)\b"
        r"|\b(?:cancel|delete)\s+(?:it|that|my visit|my booking|my appointment)\b"
        r"|\b(?:please\s+)?(?:cancel|delete)\s+(?:my\s+)?(?:site\s+)?visit\b",
        text,
    ))


def _is_cancellation_refusal(message: str) -> bool:
    """Detect an explicit refusal/negative response to an appointment cancellation confirmation."""
    text = (message or "").lower().strip()
    if not text:
        return False
    if any(p in text for p in ("no problem", "no worries")) and any(c in text for c in ("cancel", "delete")):
        return False
    refusal_keywords = (
        "no", "don't", "dont", "do not", "keep it", "keep the appointment",
        "never mind", "nevermind", "stop", "abort", "leave it",
        "keep my appointment", "keep my visit", "dont cancel", "don't cancel",
        "do not cancel", "not now", "nope", "nah", "keep the booking",
    )
    return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in refusal_keywords)


def _is_ambiguous_or_qualified_cancellation(message: str) -> bool:
    """Detect ambiguous, hesitant, or qualified responses to an appointment cancellation confirmation."""
    text = (message or "").lower().strip()
    if not text:
        return False
    ambiguous_tokens = (
        "maybe", "perhaps", "possibly", "not yet", "not sure", "i'm not sure",
        "im not sure", "wait", "hold on", "let me think", "i don't know",
        "i dont know", "thinking", "depends", "not ready", "undecided",
        "give me a minute", "give me a sec", "hang on",
    )
    if any(re.search(rf"\b{re.escape(token)}\b", text) for token in ambiguous_tokens):
        return True
    if re.search(r"\b(?:but|unless|only if|before that|first tell me|tell me first|first what)\b", text):
        return True
    if re.search(r"\b(?:what date|which date|what time|which time|when is|when was|which one|can i reschedule|can we reschedule)\b", text):
        return True
    if "?" in text and re.search(r"\b(?:cancel|appointment|booking|visit|slot|date|time|reschedule|fee|charge|why|sure)\b", text):
        return True
    return False


def _is_unambiguous_cancellation_confirmation(message: str) -> bool:
    """Detect unambiguous affirmative response to an appointment cancellation confirmation."""
    text = (message or "").lower().strip()
    if not text:
        return False
    if _is_ambiguous_or_qualified_cancellation(text):
        return False
    if _is_cancellation_refusal(text):
        return False
    norm = re.sub(r"[^\w\s]", "", text).strip()
    direct_yes = norm in {
        "yes", "yeah", "yep", "sure", "okay", "ok", "please", "yes please",
        "go ahead", "please do", "confirm", "confirmed", "cancel", "cancel it",
        "please cancel", "please cancel it", "yes cancel it", "yes please cancel it",
        "delete it", "yes delete it", "please delete it", "yes confirm", "yes go ahead",
        "definitely", "absolutely", "yes definitely", "proceed", "yes proceed",
    }
    if direct_yes:
        return True
    if re.search(r"\b(?:yes|yeah|yep|sure|please|go ahead|confirm|definitely|proceed)\b.{0,20}\b(?:cancel|delete|remove|drop)\b", text):
        return True
    if re.search(r"\b(?:cancel|delete|remove|drop)\s+(?:it|this|the appointment|the booking|the visit)\b", text):
        return True
    return False


def _is_explicit_booking_refusal(message: str) -> bool:
    """Detect an explicit refusal to create a booking, not a request to cancel a recorded lead note or existing appointment."""
    if _is_appointment_cancellation_request(message):
        return False
    text = re.sub(r"\s+", " ", (message or "").lower()).strip()
    if not text:
        return False
    return bool(re.search(
        r"\b(?:please\s+)?(?:don't|dont|do not|never|no need to|not)\b.{0,35}\b"
        r"(?:book|booking|schedule|schedule it|reserve|make a reservation|site visit)\b",
        text,
    ))


def _extract_price_evidence_line(
    chunks: List[Any],
    project_name: Optional[str],
    query: Optional[str] = None,
) -> Optional[str]:
    """Return a source line with a price and the requested unit/fee label for the named project."""
    if not project_name:
        return None
    aliases = _INFRA_PROJECT_ALIASES.get(project_name, (project_name,))
    asks_per_area_rate = any(cue in (query or "").lower() for cue in (
        "per square yard", "per sq yard", "per sq. yard", "per sq yd",
        "square yard rate", "per-square-yard",
    ))
    for chunk in chunks or []:
        content = str(getattr(chunk, "content", "") or "")
        title = str(getattr(chunk, "title", "") or "")
        doc_id = str(getattr(chunk, "doc_id", "") or "")
        if not any(alias in "\n".join((title, doc_id, content)).lower() for alias in aliases):
            continue
        for raw_line in content.splitlines():
            line = raw_line.strip(" -*#\t")
            has_amount = re.search(
                r"(?:₹|\binr\b|\brs\.?\s*)\s*[0-9][0-9,]*(?:\.\d+)?",
                line,
                re.I,
            )
            if not has_amount:
                continue
            if asks_per_area_rate:
                if re.search(
                    r"(?:per\s+(?:square|sq\.?)[-\s]*yard|per\s+sq\.?\s*yd|"
                    r"(?:square|sq\.?)[-\s]*yard\s*(?:rate|price))",
                    line,
                    re.I,
                ):
                    return line[:350]
            elif re.search(
                r"\b(price|pricing|cost|rate|booking|registration|development|charges?|fee|fees|deposit|down payment|per\s+(?:square|sq\.?)[-\s]*yard)\b",
                line,
                re.I,
            ):
                return line[:350]
    return None


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
        postgres_repo: Optional[Any] = None,
    ) -> None:
        self.llm = llm_provider
        self.knowledge_provider = knowledge_provider
        self.tool_provider = tool_provider
        self.settings = settings or get_settings()
        self.db_repo = postgres_repo
        if self.db_repo is None:
            try:
                from src.database.repository import PostgresRepository
                self.db_repo = PostgresRepository()
            except Exception:
                self.db_repo = None
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

    async def _handle_deterministic_cancellation_turn(
        self,
        state: ConversationState,
        cleaned_msg: str,
        t_start: float,
        timing: Dict[str, float],
        tracker: Optional[Any] = None,
    ) -> Optional[EngineTurnResult]:
        """Deterministically handle response to a pending appointment cancellation confirmation without cloud LLM call."""
        prior_pending_action = state.pending_action
        pending_cancellation_id = state.get_slot("cancellation_confirmation_pending_id")

        # Must have cancellation pending in state
        if not (
            prior_pending_action in ("cancel_calendar_event", "calendar_cancel_event")
            or pending_cancellation_id is not None
        ):
            return None

        # 1. Ambiguous, hesitant, or qualified response
        if _is_ambiguous_or_qualified_cancellation(cleaned_msg):
            logger.info("Deterministic cancellation: ambiguous response detected from caller: '%s'", mask_sensitive_text(cleaned_msg))
            pending_details = state.get_slot("cancellation_confirmation_pending_details")
            if pending_details:
                clarify_resp = (
                    f"Just to be certain, would you like me to cancel your appointment for {pending_details}, or keep it?"
                )
            else:
                clarify_resp = (
                    "Just to be certain, would you like me to cancel your scheduled appointment, or keep it?"
                )
            state.set_pending_action("cancel_calendar_event")
            state.set_pending_question(clarify_resp)
            state.record_turn(TurnRole.CALLER, cleaned_msg)
            state.record_turn(TurnRole.AGENT, clarify_resp)
            clarify_decision = ConversationalDecision(
                detected_domain=state.current_domain,
                detected_intent="cancel_appointment_ambiguous",
                proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                user_facing_response=clarify_resp,
                action_proposed=False,
                proposed_action=None,
                knowledge_required=False,
                needs_clarification=True,
                clarification_question=clarify_resp,
            )
            total_turn_ms = round((time.perf_counter() - t_start) * 1000, 2)
            timing["deterministic_ms"] = total_turn_ms
            return EngineTurnResult(
                response_text=clarify_resp,
                decision=clarify_decision,
                conversation_active=True,
                termination_occurred=False,
                latency_ms=total_turn_ms,
                latency_breakdown=timing,
            )

        # 2. Refusal / Negative response
        if _is_cancellation_refusal(cleaned_msg):
            logger.info("Deterministic cancellation: refusal detected from caller; preserving appointment")
            state.clear_slot("cancellation_confirmation_pending_id")
            state.clear_slot("cancellation_confirmation_pending_details")
            state.set_pending_action(None)
            state.set_pending_question(None)
            keep_response = "Understood. I will keep your scheduled appointment intact. Is there anything else I can help you with?"
            state.record_turn(TurnRole.CALLER, cleaned_msg)
            state.record_turn(TurnRole.AGENT, keep_response)
            keep_decision = ConversationalDecision(
                detected_domain=state.current_domain,
                detected_intent="cancel_appointment_aborted",
                proposed_stage=ConversationStage.INFORMATION,
                user_facing_response=keep_response,
                action_proposed=False,
                proposed_action=None,
                knowledge_required=False,
            )
            total_turn_ms = round((time.perf_counter() - t_start) * 1000, 2)
            timing["deterministic_ms"] = total_turn_ms
            return EngineTurnResult(
                response_text=keep_response,
                decision=keep_decision,
                conversation_active=True,
                termination_occurred=False,
                latency_ms=total_turn_ms,
                latency_breakdown=timing,
            )

        # 3. Affirmative response
        if _is_unambiguous_cancellation_confirmation(cleaned_msg):
            # Check 3A: Inconsistent / Missing state -> fail closed
            caller_phone = state.caller.phone or state.get_slot("phone")
            valid_id = False
            appt_id_int: Optional[int] = None
            if pending_cancellation_id is not None:
                try:
                    appt_id_int = int(pending_cancellation_id)
                    if appt_id_int > 0:
                        valid_id = True
                except (ValueError, TypeError):
                    valid_id = False

            if not valid_id or not caller_phone or appt_id_int is None:
                logger.warning(
                    "Deterministic cancellation: inconsistent/invalid pending state (id=%s, phone=%s); failing closed",
                    pending_cancellation_id,
                    mask_phone_number(caller_phone) if caller_phone else "None",
                )
                state.clear_slot("cancellation_confirmation_pending_id")
                state.clear_slot("cancellation_confirmation_pending_details")
                state.set_pending_action(None)
                state.set_pending_question(None)
                fail_closed_resp = (
                    "I'm unable to verify the appointment details for cancellation. "
                    "No appointments have been modified. Would you like me to look up your scheduled visit again?"
                )
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, fail_closed_resp)
                fail_closed_dec = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="cancel_appointment_invalid_state",
                    proposed_stage=ConversationStage.INFORMATION,
                    user_facing_response=fail_closed_resp,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                total_turn_ms = round((time.perf_counter() - t_start) * 1000, 2)
                timing["deterministic_ms"] = total_turn_ms
                return EngineTurnResult(
                    response_text=fail_closed_resp,
                    decision=fail_closed_dec,
                    conversation_active=True,
                    termination_occurred=False,
                    latency_ms=total_turn_ms,
                    latency_breakdown=timing,
                )

            # Check 3B: Sensitive Authorization Gate (Phase 3E)
            caller_verified = (
                state.get_slot("returning_caller_verified_sensitive", False)
                or getattr(state.caller, "verified_sensitive", False) is True
                or state.get_slot("caller_verified_sensitive", False)
            )
            if not caller_verified:
                logger.warning("Deterministic cancellation: unverified caller attempting cancellation; rejecting")
                state.clear_slot("cancellation_confirmation_pending_id")
                state.clear_slot("cancellation_confirmation_pending_details")
                state.set_pending_action(None)
                state.set_pending_question(None)
                auth_resp = (
                    "For security, cancelling or deleting an appointment requires verified caller authentication. "
                    "Please contact customer support to verify your identity before cancelling."
                )
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, auth_resp)
                auth_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="cancel_appointment_unauthorized",
                    proposed_stage=ConversationStage.INFORMATION,
                    user_facing_response=auth_resp,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                total_turn_ms = round((time.perf_counter() - t_start) * 1000, 2)
                timing["deterministic_ms"] = total_turn_ms
                return EngineTurnResult(
                    response_text=auth_resp,
                    decision=auth_decision,
                    conversation_active=True,
                    termination_occurred=False,
                    latency_ms=total_turn_ms,
                    latency_breakdown=timing,
                )

            # Check 3C: Tool Provider Available
            if not self.tool_provider:
                logger.warning("Deterministic cancellation: tool provider unavailable")
                state.clear_slot("cancellation_confirmation_pending_id")
                state.clear_slot("cancellation_confirmation_pending_details")
                state.set_pending_action(None)
                state.set_pending_question(None)
                unavail_resp = (
                    "I can't complete that cancellation request right now because the required service is unavailable. "
                    "The existing appointment was retained in our records."
                )
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, unavail_resp)
                unavail_dec = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="cancel_appointment_service_unavailable",
                    proposed_stage=ConversationStage.INFORMATION,
                    user_facing_response=unavail_resp,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                total_turn_ms = round((time.perf_counter() - t_start) * 1000, 2)
                timing["deterministic_ms"] = total_turn_ms
                return EngineTurnResult(
                    response_text=unavail_resp,
                    decision=unavail_dec,
                    conversation_active=True,
                    termination_occurred=False,
                    latency_ms=total_turn_ms,
                    latency_breakdown=timing,
                )

            # Execute Tool through existing tool provider dispatcher
            tool_name = "cancel_calendar_event"
            tool_args = {
                "appointment_id": appt_id_int,
                "phone_number": caller_phone,
                "confirmed": True,
                "caller_verified_sensitive": True,
                "returning_caller": state.get_slot("returning_caller", False),
                "domain": state.current_domain.value,
            }
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
            tool_duration = round((time.perf_counter() - t_tool) * 1000, 2)
            timing["tool_execution_ms"] = tool_duration
            if tracker:
                tracker.mark_tool_end()

            proposed_action = ProposedAction(
                tool_name=tool_name,
                arguments=tool_args,
            )

            if tool_exec_result.success:
                state.complete_action(tool_name, tool_exec_result)
                state.clear_slot("cancellation_confirmation_pending_id")
                state.clear_slot("cancellation_confirmation_pending_details")
                state.set_pending_question(None)
                speech_confirmation = (
                    "Your appointment has been successfully cancelled. "
                    "The database record and calendar slot have been removed. Is there anything else I can assist you with?"
                )
                final_stage = ConversationStage.COMPLETED
            else:
                state.set_pending_action(None)
                state.last_tool_result = tool_exec_result
                state.update_slot("cancellation_confirmation_pending_id", None)
                state.update_slot("cancellation_confirmation_pending_details", None)
                speech_confirmation = (
                    "I couldn't cancel that appointment due to a system issue. "
                    "The existing appointment was retained in our records. Would you like to try again?"
                )
                final_stage = ConversationStage.INFORMATION

            state.record_turn(TurnRole.CALLER, cleaned_msg)
            state.record_turn(TurnRole.AGENT, speech_confirmation)

            decision = ConversationalDecision(
                detected_domain=state.current_domain,
                detected_intent="site_visit_cancellation",
                proposed_stage=final_stage,
                user_facing_response=speech_confirmation,
                action_proposed=True,
                proposed_action=proposed_action,
                knowledge_required=False,
            )
            total_turn_ms = round((time.perf_counter() - t_start) * 1000, 2)
            timing["total_turn_ms"] = total_turn_ms
            return EngineTurnResult(
                response_text=speech_confirmation,
                decision=decision,
                action_proposed=True,
                proposed_action=proposed_action,
                conversation_active=True,
                termination_occurred=False,
                tool_result=tool_exec_result,
                latency_ms=total_turn_ms,
                latency_breakdown=timing,
            )

        # 4. Caller responded with something else (e.g. unrelated question)
        # Fall through to normal conversational processing pipeline.
        return None

    @staticmethod
    async def _invoke_grounded_callback(
        callback: Callable[[str], Any],
        phrase: str,
    ) -> None:
        """Safely invoke streaming phrase callback, supporting both async and sync callables."""
        res = callback(phrase)
        if asyncio.iscoroutine(res):
            await res

    async def process_user_turn(
        self,
        state: ConversationState,
        user_message: str,
        tracker: Optional[Any] = None,
        on_grounded_chunk: Optional[Callable[[str], Any]] = None,
    ) -> EngineTurnResult:
        """Process an incoming caller utterance and advance conversation state."""
        cleaned_msg = user_message.strip()

        # Outbound identity must be verified before any RAG or external-action flow.
        identity_result = await self._handle_outbound_identity_gate(state, cleaned_msg)
        if identity_result is not None:
            return identity_result

        # 1. Check for immediate explicit termination from user utterance
        if ConversationState.is_explicit_termination(cleaned_msg):
            logger.info("Explicit termination signal detected from caller: '%s'", mask_sensitive_text(cleaned_msg))
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

        # Deterministic Cancellation Confirmation Path (Phase 5B-1)
        cancellation_result = await self._handle_deterministic_cancellation_turn(
            state=state,
            cleaned_msg=cleaned_msg,
            t_start=t_start,
            timing=timing,
            tracker=tracker,
        )
        if cancellation_result is not None:
            if tracker:
                tracker.mark_response_ready()
            return cancellation_result

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
                if not decision.knowledge_required and decision.user_facing_response in ("Hi, how can I help you today?", None, ""):
                    decision.user_facing_response = "Hello! Welcome to Infra Builder. How can I assist you with your property search today?"
        else:
            # Outbound calls: ensure active domain is Infra Builder
            state.current_domain = DomainType.INFRA_BUILDER
            decision.detected_domain = DomainType.INFRA_BUILDER

        # Capture pending question/action prior to intent preemption.
        prior_pending_question = state.pending_question
        prior_pending_action = state.pending_action
        previous_agent_text = next(
            (turn.content for turn in reversed(state.history) if turn.role == TurnRole.AGENT),
            "",
        )
        normalized_current_reply = self._normalize_confirmation_text(cleaned_msg)
        followup_offer_was_pending = any(
            "follow-up" in (text or "").lower()
            or "follow up" in (text or "").lower()
            or "callback" in (text or "").lower()
            for text in (prior_pending_question, previous_agent_text)
        ) and any(
            phrase in (text or "").lower()
            for text in (prior_pending_question, previous_agent_text)
            for phrase in ("would you like", "can i", "shall i", "do you want me to")
        )
        followup_acceptance = (
            normalized_current_reply in {
                "yes", "yeah", "yep", "sure", "okay", "ok", "please", "yes please",
                "go ahead", "please do", "that would be great",
            }
            and followup_offer_was_pending
        )
        followup_requested = (
            state.current_domain == DomainType.INFRA_BUILDER
            and (
                _is_explicit_project_followup_request(cleaned_msg)
                or followup_acceptance
            )
        )

        # A new explicit request supersedes the previous pending flow. The pricing
        # fallback must never swallow a request to record a follow-up.
        if followup_requested:
            state.set_pending_action(None)
            state.set_pending_question(None)
            decision.detected_intent = "request_project_team_followup"
            decision.proposed_stage = ConversationStage.ACTION_CONFIRMATION
            decision.knowledge_required = False
            decision.knowledge_query = None
            decision.needs_clarification = False
            decision.clarification_question = None
            caller_name_for_followup = (
                (state.caller.name or state.get_slot("caller_name") or state.contact_name or "").strip()
            )
            caller_phone_for_followup = (
                state.caller.phone or state.get_slot("phone") or state.get_slot("whatsapp_number")
            )
            project_for_followup = (
                _project_from_conversation_context(
                    state.history,
                    cleaned_msg,
                    preferred_project=state.get_slot("preferred_project"),
                )
                or "property pricing"
            )
            followup_note = (
                f"Customer requested project-team follow-up to confirm current pricing. "
                f"Project/topic: {project_for_followup}. Caller message: {cleaned_msg}"
            )
            if caller_name_for_followup:
                decision.action_proposed = True
                decision.proposed_action = ProposedAction(
                    tool_name="update_lead",
                    arguments={
                        "name": caller_name_for_followup,
                        "phone": caller_phone_for_followup,
                        "status": "follow_up_requested",
                        "notes": followup_note,
                        "interested_in": project_for_followup,
                        "follow_up_reason": followup_note,
                        "domain": DomainType.INFRA_BUILDER.value,
                    },
                    confidence=1.0,
                )
                decision.user_facing_response = (
                    "I'll record your request for the project team to confirm the current price."
                )
                state.update_slot("project_followup_pending", False)
            else:
                decision.action_proposed = False
                decision.proposed_action = None
                decision.user_facing_response = (
                    "I can record that follow-up request, but first, may I have your name?"
                )
                state.update_slot("project_followup_pending", True)
                state.set_pending_action("update_lead")
                state.set_pending_question("May I have your name so I can record the follow-up request?")
        else:
            decision.action_proposed = bool(decision.action_proposed and decision.proposed_action)

        # 6. Intent Preemption & Updating
        # The latest explicit user intent has priority over prior pending questions
        if decision.detected_intent:
            state.set_intent(
                new_intent=decision.detected_intent,
                sub_intent=decision.detected_sub_intent,
                clear_pending_question=True,
            )
        if state.get_slot("project_followup_pending") is True:
            state.set_pending_action("update_lead")
            state.set_pending_question("May I have your name so I can record the follow-up request?")

        # Split site-visit date and time instead of sending a combined phrase such as
        # "tomorrow 10 am" to the calendar availability lookup as if it were a date.
        visit_date, visit_time = _extract_site_visit_date_time(cleaned_msg)
        extracted_pref = extract_datetime_preference(cleaned_msg)
        if visit_date:
            extracted_pref = visit_date
            decision.extracted_slots["meeting_preference"] = visit_date
        elif visit_time and state.get_slot("preferred_date"):
            # A time-only reply (e.g. "10 am") keeps the already-known date.
            extracted_pref = state.get_slot("preferred_date")
        elif extracted_pref:
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
            if (
                state.get_slot("returning_caller") is True
                and not state.get_slot("returning_caller_verified_sensitive", False)
                and state.caller.name
                and extracted_spoken_name.strip().lower() != state.caller.name.strip().lower()
            ):
                logger.warning(
                    "Caller privacy guard: blocked unverified returning caller from overwriting registered profile name (call_id=%s)",
                    state.metadata.call_id,
                )
            else:
                state.caller.name = extracted_spoken_name
                state.update_slot("caller_name", extracted_spoken_name, sync_caller=True)
                # Keep legacy aliases synchronized if they were produced by the LLM.
                for alias in ("name", "contact_name"):
                    if alias in decision.extracted_slots:
                        decision.extracted_slots[alias] = extracted_spoken_name
                decision.extracted_slots["caller_name"] = extracted_spoken_name
                logger.info(
                    "Updated caller name from caller utterance (call_id=%s)",
                    state.metadata.call_id,
                )

        if state.get_slot("project_followup_pending") is True:
            pending_followup_name = (
                (state.caller.name or state.get_slot("caller_name") or state.contact_name or "").strip()
            )
            if pending_followup_name:
                pending_followup_phone = (
                    state.caller.phone or state.get_slot("phone") or state.get_slot("whatsapp_number")
                )
                project_for_followup = (
                    _project_from_conversation_context(
                        state.history,
                        cleaned_msg,
                        preferred_project=state.get_slot("preferred_project"),
                    )
                    or "property pricing"
                )
                followup_note = (
                    f"Customer requested project-team follow-up to confirm current pricing. "
                    f"Project/topic: {project_for_followup}. Original request is in the conversation history."
                )
                decision.detected_intent = "request_project_team_followup"
                decision.proposed_stage = ConversationStage.ACTION_CONFIRMATION
                decision.knowledge_required = False
                decision.knowledge_query = None
                decision.action_proposed = True
                decision.proposed_action = ProposedAction(
                    tool_name="update_lead",
                    arguments={
                        "name": pending_followup_name,
                        "phone": pending_followup_phone,
                        "status": "follow_up_requested",
                        "notes": followup_note,
                        "interested_in": project_for_followup,
                        "follow_up_reason": followup_note,
                        "domain": DomainType.INFRA_BUILDER.value,
                    },
                    confidence=1.0,
                )
                decision.user_facing_response = (
                    "I'll record your request for the project team to confirm the current price."
                )
                state.update_slot("project_followup_pending", False)
                state.set_pending_question(None)

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

        if visit_date:
            state.update_slot("preferred_date", visit_date, sync_caller=True)
            state.update_slot("meeting_preference", visit_date, sync_caller=True)
        if visit_time:
            state.update_slot("preferred_time", visit_time, sync_caller=True)
        if extracted_pref and not visit_date and not (visit_time and state.get_slot("preferred_date")) and (
            not state.get_slot("meeting_preference")
            or any(w in cleaned_msg.lower() for w in ["actually", "instead", "make it", "change", "rather"])
        ):
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
                        (_slot_key(candidate) in caller_slot_key or caller_slot_key in _slot_key(candidate))
                        or (visit_time and _slot_matches_requested_time(candidate, visit_time))
                    )
                    and len(caller_slot_key) >= 2
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

        # Appointment Cancellation Flow
        is_cancellation_request = _is_appointment_cancellation_request(cleaned_msg)
        is_explicit_confirmation = _is_explicit_action_confirmation(cleaned_msg)
        pending_cancellation_id = state.get_slot("cancellation_confirmation_pending_id")
        is_cancel_refusal = any(w in cleaned_msg.lower() for w in ["no", "don't", "dont", "keep it", "never mind", "stop", "abort", "leave it"])
        norm_reply = self._normalize_confirmation_text(cleaned_msg)
        is_cancellation_confirmed = (
            is_explicit_confirmation
            or norm_reply in {"yes", "yeah", "yep", "sure", "okay", "ok", "please", "yes please", "go ahead", "please do", "confirm", "confirmed"}
            or bool(re.search(r"\b(?:yes|yeah|yep|sure|please|go ahead|confirm)\b.{0,20}\b(?:cancel|delete|remove|drop)\b", cleaned_msg.lower()))
            or bool(re.search(r"\b(?:cancel|delete|remove|drop)\s+(?:it|this|the appointment|the booking|the visit)\b", cleaned_msg.lower()))
        )

        if pending_cancellation_id and prior_pending_action in ("cancel_calendar_event", "calendar_cancel_event"):
            caller_phone = state.caller.phone or state.get_slot("phone")
            if is_cancel_refusal:
                state.clear_slot("cancellation_confirmation_pending_id")
                state.clear_slot("cancellation_confirmation_pending_details")
                state.set_pending_action(None)
                keep_response = "Understood. I will keep your scheduled appointment intact. Is there anything else I can help you with?"
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, keep_response)
                keep_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="cancel_appointment_aborted",
                    proposed_stage=ConversationStage.INFORMATION,
                    user_facing_response=keep_response,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                return EngineTurnResult(
                    response_text=keep_response,
                    decision=keep_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )
            elif is_cancellation_confirmed:
                decision.action_proposed = True
                decision.proposed_action = ProposedAction(
                    tool_name="cancel_calendar_event",
                    arguments={
                        "appointment_id": pending_cancellation_id,
                        "phone_number": caller_phone,
                        "confirmed": True,
                        "caller_verified_sensitive": (
                            state.get_slot("returning_caller_verified_sensitive", False)
                            or getattr(state.caller, "verified_sensitive", False)
                            or state.get_slot("caller_verified_sensitive", False)
                        ),
                        "returning_caller": state.get_slot("returning_caller", False),
                    },
                )
                action_proposed = True

        elif is_cancellation_request:
            caller_phone = state.caller.phone or state.get_slot("phone")
            is_verified = (
                state.get_slot("returning_caller_verified_sensitive", False)
                or getattr(state.caller, "verified_sensitive", False) is True
                or state.get_slot("caller_verified_sensitive", False)
            )

            # Security Authorization Gate: Sensitive deletion requires verified caller identity
            if not is_verified:
                auth_resp = (
                    "For security, cancelling or deleting an appointment requires verified caller authentication. "
                    "Please contact customer support to verify your identity before cancelling."
                )
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, auth_resp)
                auth_decision = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="cancel_appointment_unauthorized",
                    proposed_stage=ConversationStage.INFORMATION,
                    user_facing_response=auth_resp,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                return EngineTurnResult(
                    response_text=auth_resp,
                    decision=auth_decision,
                    conversation_active=True,
                    termination_occurred=False,
                )

            # Retrieve caller appointments from PostgreSQL
            appts: List[Dict[str, Any]] = []
            if self.db_repo and caller_phone:
                try:
                    appts = await self.db_repo.get_appointments_by_phone(caller_phone)
                except Exception as repo_err:
                    logger.warning("Error querying appointments for cancellation: %s", repo_err)

            if not appts:
                no_appts_resp = "I checked our records, but I couldn't find any scheduled appointments under your phone number."
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, no_appts_resp)
                no_appts_dec = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="cancel_appointment_none_found",
                    proposed_stage=ConversationStage.INFORMATION,
                    user_facing_response=no_appts_resp,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                return EngineTurnResult(
                    response_text=no_appts_resp,
                    decision=no_appts_dec,
                    conversation_active=True,
                    termination_occurred=False,
                )

            elif len(appts) == 1:
                target_appt = appts[0]
                t_id = target_appt["id"]
                t_proj = target_appt.get("project_name") or target_appt.get("title") or "the property"
                t_date = target_appt.get("event_date") or "the scheduled date"
                t_time = target_appt.get("event_time") or "the scheduled time"
                confirm_prompt = f"You have a site visit for {t_proj} scheduled on {t_date} at {t_time}. Are you sure you would like to cancel this appointment?"
                state.set_pending_action("cancel_calendar_event")
                state.update_slot("cancellation_confirmation_pending_id", t_id)
                state.update_slot("cancellation_confirmation_pending_details", f"{t_proj} on {t_date} at {t_time}")
                state.set_pending_question(confirm_prompt)
                state.record_turn(TurnRole.CALLER, cleaned_msg)
                state.record_turn(TurnRole.AGENT, confirm_prompt)
                one_cand_dec = ConversationalDecision(
                    detected_domain=state.current_domain,
                    detected_intent="site_visit_cancellation",
                    proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                    user_facing_response=confirm_prompt,
                    action_proposed=False,
                    proposed_action=None,
                    knowledge_required=False,
                )
                return EngineTurnResult(
                    response_text=confirm_prompt,
                    decision=one_cand_dec,
                    conversation_active=True,
                    termination_occurred=False,
                )

            else:
                matched = [
                    a for a in appts
                    if (a.get("project_name") and a.get("project_name").lower() in cleaned_msg.lower())
                    or (a.get("event_date") and a.get("event_date").lower() in cleaned_msg.lower())
                    or (a.get("event_time") and a.get("event_time").lower() in cleaned_msg.lower())
                ]
                if len(matched) == 1:
                    target_appt = matched[0]
                    t_id = target_appt["id"]
                    t_proj = target_appt.get("project_name") or target_appt.get("title") or "the property"
                    t_date = target_appt.get("event_date") or "the scheduled date"
                    t_time = target_appt.get("event_time") or "the scheduled time"
                    confirm_prompt = f"You have a site visit for {t_proj} scheduled on {t_date} at {t_time}. Are you sure you would like to cancel this appointment?"
                    state.set_pending_action("cancel_calendar_event")
                    state.update_slot("cancellation_confirmation_pending_id", t_id)
                    state.update_slot("cancellation_confirmation_pending_details", f"{t_proj} on {t_date} at {t_time}")
                    state.set_pending_question(confirm_prompt)
                    state.record_turn(TurnRole.CALLER, cleaned_msg)
                    state.record_turn(TurnRole.AGENT, confirm_prompt)
                    match_dec = ConversationalDecision(
                        detected_domain=state.current_domain,
                        detected_intent="site_visit_cancellation",
                        proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                        user_facing_response=confirm_prompt,
                        action_proposed=False,
                        proposed_action=None,
                        knowledge_required=False,
                    )
                    return EngineTurnResult(
                        response_text=confirm_prompt,
                        decision=match_dec,
                        conversation_active=True,
                        termination_occurred=False,
                    )
                else:
                    items = "; ".join(f"{a.get('project_name') or a.get('title')} on {a.get('event_date')} at {a.get('event_time')}" for a in appts[:3])
                    disambig_prompt = f"You have multiple appointments scheduled: {items}. Which appointment would you like to cancel?"
                    state.set_pending_action("cancel_calendar_event")
                    state.set_pending_question(disambig_prompt)
                    state.record_turn(TurnRole.CALLER, cleaned_msg)
                    state.record_turn(TurnRole.AGENT, disambig_prompt)
                    multi_dec = ConversationalDecision(
                        detected_domain=state.current_domain,
                        detected_intent="site_visit_cancellation",
                        proposed_stage=ConversationStage.ACTION_CONFIRMATION,
                        user_facing_response=disambig_prompt,
                        action_proposed=False,
                        proposed_action=None,
                        knowledge_required=False,
                    )
                    return EngineTurnResult(
                        response_text=disambig_prompt,
                        decision=multi_dec,
                        conversation_active=True,
                        termination_occurred=False,
                    )

        # Promote to calendar action if caller is requesting scheduling or answering pending date/time question
        is_calendar_request = (
            any(
                w in cleaned_msg.lower()
                for w in ["schedule", "consultation", "demo", "calendar", "book a slot", "site visit", "visit", "tour"]
            )
            and not _is_appointment_cancellation_request(cleaned_msg)
        )

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
            if not final_response_text or not final_response_text.strip():
                final_response_text = "How can I help you today?"
                decision.user_facing_response = final_response_text

        # Simple conversational acknowledgments do not require factual RAG retrieval
        simple_acks = {
            "hello", "hi", "hey", "yes", "yeah", "yep", "sure", "okay", "ok",
            "thanks", "thank you", "great", "fine", "alright", "bye", "goodbye",
            "sounds good", "perfect", "no problem", "correct",
        }
        if cleaned_msg.lower().strip() in simple_acks:
            decision.knowledge_required = False
            decision.knowledge_query = None
            if not final_response_text or not final_response_text.strip():
                final_response_text = "Understood. How else can I assist you with your property search?"
                decision.user_facing_response = final_response_text

        # Portfolio-wide project requests must not depend on the LLM remembering
        # to set knowledge_required. Detect them deterministically and force the
        # existing catalog retrieval path for the locked Infra Builder domain.
        catalog_requested = bool(
            state.current_domain == DomainType.INFRA_BUILDER
            and not followup_requested
            and is_catalog_query(cleaned_msg, state.current_domain)
        )
        if catalog_requested:
            decision.knowledge_required = True
            decision.knowledge_query = cleaned_msg
            logger.info(
                "Forcing Infra Builder catalog retrieval for caller query: %s",
                cleaned_msg,
            )


        # Pricing and fee claims require explicit retrieved evidence; never trust the
        # first-pass LLM response as proof of a rate or of an "official" source.
        pricing_meta_requested = bool(
            state.current_domain == DomainType.INFRA_BUILDER
            and _is_infra_pricing_meta_query(cleaned_msg)
            and not followup_requested
        )
        pricing_requested = bool(
            state.current_domain == DomainType.INFRA_BUILDER
            and _is_infra_pricing_query(cleaned_msg)
            and not pricing_meta_requested
            and not followup_requested
        )
        pricing_project = _project_from_conversation_context(
            state.history,
            cleaned_msg,
            preferred_project=state.get_slot("preferred_project"),
        ) if (pricing_requested or pricing_meta_requested) else None
        if pricing_meta_requested:
            decision.knowledge_required = True
            decision.knowledge_query = cleaned_msg
            logger.info(
                "Forcing Infra Builder pricing-documentation retrieval (project=%s): %s",
                pricing_project or "unspecified",
                cleaned_msg,
            )
        if pricing_requested:
            decision.knowledge_required = True
            decision.knowledge_query = cleaned_msg
            logger.info(
                "Forcing Infra Builder pricing retrieval (project=%s) for caller query: %s",
                pricing_project or "unspecified",
                cleaned_msg,
            )

        if decision.knowledge_required and not self.knowledge_provider:
            logger.warning("Knowledge was required but no knowledge provider is configured.")
            if pricing_meta_requested:
                final_response_text = (
                    "I can't inspect the knowledge base right now, so I can't confirm what pricing information "
                    "is stored there. I don't want to guess."
                )
            elif pricing_requested:
                final_response_text = (
                    f"I can't verify the current price or charges for {pricing_project.title() if pricing_project else 'that project'} "
                    "because the knowledge base isn't configured. Would you like me to record a follow-up request for the project team?"
                )
            else:
                final_response_text = (
                    "I don't have verified property details available right now. "
                    "I can help with another question or record a follow-up request."
                )
            decision.user_facing_response = final_response_text
            decision.knowledge_required = False
            decision.knowledge_query = None

        if decision.knowledge_required and self.knowledge_provider:
            t_rag = time.perf_counter()
            if tracker:
                tracker.mark_rag_start()
            query_text = cleaned_msg if (catalog_requested or pricing_requested) else (
                decision.knowledge_query or decision.detected_intent or cleaned_msg
            )
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
                    if pricing_meta_requested:
                        final_response_text = (
                            "I couldn't access the knowledge base just now, so I can't verify which pricing details "
                            "are stored there."
                        )
                    elif pricing_requested:
                        final_response_text = (
                            f"I couldn't retrieve a verified price for {pricing_project.title() if pricing_project else 'that project'} "
                            "because the knowledge base is unavailable. Would you like me to record a follow-up request for the project team?"
                        )
                    else:
                        final_response_text = (
                            "Our knowledge base is currently unavailable, so I can't verify those property details right now. "
                            "I can record a follow-up request if you'd like."
                        )
                    decision.user_facing_response = final_response_text
                elif not grounded_res.knowledge_available or not grounded_res.chunks:
                    logger.info("No verified knowledge passed relevance threshold for query '%s'", query_text)
                    if pricing_meta_requested:
                        final_response_text = (
                            "I couldn't retrieve a relevant pricing entry from the knowledge base. That doesn't prove "
                            "no price exists elsewhere; it means I can't verify one from the records retrieved."
                        )
                    elif pricing_requested:
                        final_response_text = (
                            f"I couldn't retrieve a verified price for {pricing_project.title() if pricing_project else 'that project'}. "
                            "That doesn't prove a price is unavailable; I just can't verify it from the records retrieved. "
                            "Would you like me to record a follow-up request for the project team?"
                        )
                    else:
                        final_response_text = (
                            "I checked our knowledge base, but I don't have verified details regarding that at the moment. "
                            "Would you like me to record a follow-up request, or is there another question I can answer?"
                        )
                    decision.user_facing_response = final_response_text
                elif pricing_meta_requested:
                    grounded_citations = [c.doc_id for c in grounded_res.chunks]
                    if not pricing_project:
                        final_response_text = (
                            "Which Infra Builder project do you mean? I can check the retrieved knowledge-base records "
                            "for that project's pricing information."
                        )
                    else:
                        has_price_entry = _has_verified_price_evidence(
                            grounded_res.chunks,
                            pricing_project,
                            cleaned_msg,
                        )
                        evidence_line = _extract_price_evidence_line(
                            grounded_res.chunks,
                            pricing_project,
                        )
                        if has_price_entry and evidence_line:
                            final_response_text = (
                                f"The retrieved knowledge-base content contains this pricing entry for "
                                f"{pricing_project.title()}: {evidence_line}"
                            )
                        elif has_price_entry:
                            final_response_text = (
                                f"The retrieved records appear to contain a pricing amount for {pricing_project.title()}, "
                                "but I couldn't isolate a clear source line and unit to quote safely. I won't guess at the rate."
                            )
                        else:
                            final_response_text = (
                                f"In the knowledge-base content retrieved for {pricing_project.title()}, I couldn't find "
                                "a verified price or per-square-yard rate. I can only speak to the records retrieved, "
                                "not guarantee that no price exists in another document."
                            )
                    decision.user_facing_response = final_response_text
                elif pricing_requested and not _has_verified_price_evidence(
                    grounded_res.chunks,
                    pricing_project,
                    cleaned_msg,
                ):
                    # Fail closed before any LLM synthesis: relevant property context is
                    # not enough to prove a specific price or fee.
                    grounded_citations = [c.doc_id for c in grounded_res.chunks]
                    if not pricing_project:
                        final_response_text = (
                            "Which Infra Builder project would you like pricing for? "
                            "I need the specific project before I can check its verified rate."
                        )
                    else:
                        final_response_text = (
                            f"I can't verify the current price or charges for {pricing_project.title()} "
                            "from the project information available to me. I don't want to give you "
                            "an unconfirmed rate. Would you like me to record a follow-up request for the project team to confirm it?"
                        )
                    decision.user_facing_response = final_response_text
                    logger.warning(
                        "Blocked unsupported Infra Builder price claim: project=%s query=%s",
                        pricing_project or "unspecified",
                        cleaned_msg,
                    )
                else:
                    # Chunks found and passed relevance threshold
                    grounded_citations = [c.doc_id for c in grounded_res.chunks]

                    if pricing_requested:
                        evidence_line = _extract_price_evidence_line(
                            grounded_res.chunks,
                            pricing_project,
                            cleaned_msg,
                        )
                        if evidence_line:
                            final_response_text = (
                                f"The retrieved project information for {pricing_project.title()} states: "
                                f"{evidence_line}"
                            )
                        else:
                            final_response_text = (
                                f"I found related information for {pricing_project.title()}, but couldn't isolate a "
                                "source line that explicitly confirms the requested price and unit. I won't guess at the rate."
                            )
                        decision.user_facing_response = final_response_text
                    else:
                        grounding_prompt = render_prompt(
                            "rag/grounded_answer.txt",
                            caller_message=cleaned_msg,
                            domain=decision.detected_domain.value,
                            formatted_context=grounded_res.formatted_context,
                        ).strip()
                        system_instruction = load_prompt("rag/system.txt").strip()
                        t_ground = time.perf_counter()
                        if tracker:
                            tracker.mark_grounded_synth_start(t_ground)

                        use_streaming = (
                            on_grounded_chunk is not None
                            and hasattr(self.llm, "stream_response")
                        )

                        if use_streaming:
                            from src.voice.streaming_buffer import StreamingPhraseBuffer

                            buffer = StreamingPhraseBuffer()
                            accumulated_fragments: List[str] = []
                            emitted_phrases: List[str] = []

                            try:
                                stream_iter = self.llm.stream_response(
                                    prompt=grounding_prompt,
                                    system_instruction=system_instruction,
                                )
                                async for fragment in stream_iter:
                                    accumulated_fragments.append(fragment)
                                    phrases = buffer.feed(fragment)
                                    for phrase in phrases:
                                        if phrase.strip():
                                            if tracker:
                                                tracker.mark_first_grounded_phrase()
                                            emitted_phrases.append(phrase)
                                            try:
                                                await self._invoke_grounded_callback(on_grounded_chunk, phrase)
                                            except asyncio.CancelledError:
                                                raise
                                            except Exception as cb_err:
                                                raise StreamingCallbackError(f"Callback failed: {cb_err}") from cb_err

                                trailing_phrases = buffer.flush()
                                for phrase in trailing_phrases:
                                    if phrase.strip():
                                        if tracker:
                                            tracker.mark_first_grounded_phrase()
                                        emitted_phrases.append(phrase)
                                        try:
                                            await self._invoke_grounded_callback(on_grounded_chunk, phrase)
                                        except asyncio.CancelledError:
                                            raise
                                        except Exception as cb_err:
                                            raise StreamingCallbackError(f"Callback failed: {cb_err}") from cb_err

                                full_text = "".join(accumulated_fragments).strip()
                                if not full_text and emitted_phrases:
                                    full_text = " ".join(emitted_phrases).strip()
                                final_response_text = full_text
                                if tracker:
                                    tracker.mark_grounded_gen_end()

                            except asyncio.CancelledError:
                                logger.info("Grounded RAG streaming cancelled by caller barge-in.")
                                raise
                            except StreamingCallbackError:
                                final_response_text = " ".join(emitted_phrases).strip()
                                decision.user_facing_response = final_response_text
                                raise
                            except Exception as stream_err:
                                if not emitted_phrases:
                                    logger.warning(
                                        "Grounded RAG streaming failed before any text was emitted (%s); falling back to batch.",
                                        stream_err,
                                    )
                                    grounded_speech = await self.llm.generate_response(
                                        prompt=grounding_prompt,
                                        system_instruction=system_instruction,
                                    )
                                    final_response_text = grounded_speech.strip()
                                    if tracker:
                                        tracker.mark_first_grounded_phrase()
                                        tracker.mark_grounded_gen_end()
                                else:
                                    logger.warning(
                                        "Grounded RAG streaming failed mid-stream after emitting %d phrases (%s); preserving partial response.",
                                        len(emitted_phrases),
                                        stream_err,
                                    )
                                    final_response_text = " ".join(emitted_phrases).strip()
                                    if tracker:
                                        tracker.mark_grounded_gen_end()
                        else:
                            grounded_speech = await self.llm.generate_response(
                                prompt=grounding_prompt,
                                system_instruction=system_instruction,
                            )
                            final_response_text = grounded_speech.strip()
                            if tracker:
                                tracker.mark_first_grounded_phrase()
                                tracker.mark_grounded_gen_end()

                        timing["grounded_llm_ms"] = round((time.perf_counter() - t_ground) * 1000, 2)
                        # A portfolio answer is incomplete unless it names every
                        # project in the verified catalog. Fall back to a safe
                        # names-only answer rather than voicing another generic
                        # response or inventing project details.
                        if catalog_requested:
                            required_project_names = (
                                "jb pride",
                                "nature valley",
                                "pristine city",
                                "serene city",
                            )
                            response_lower = final_response_text.lower()
                            missing_projects = [
                                name for name in required_project_names
                                if name not in response_lower
                            ]
                            if missing_projects:
                                logger.warning(
                                    "Grounded catalog answer omitted projects %s; using verified names-only fallback.",
                                    missing_projects,
                                )
                                final_response_text = (
                                    "Our projects are JB Pride, Nature Valley, "
                                    "Pristine City, and Serene City. Which project "
                                    "would you like to know more about?"
                                )

                        decision.user_facing_response = final_response_text
            except Exception as rag_err:
                if isinstance(rag_err, StreamingCallbackError):
                    raise rag_err.__cause__ or rag_err
                timing["rag_retrieval_ms"] = round((time.perf_counter() - t_rag) * 1000, 2)
                if tracker:
                    tracker.mark_rag_end()
                logger.error("RAG retrieval failed: %s", rag_err)
                final_response_text = (
                    "I couldn't retrieve verified property details because the knowledge base returned an error. "
                    "I haven't arranged a follow-up. Would you like me to record a follow-up request for the project team?"
                )
                decision.user_facing_response = final_response_text

        # Safety fallback: ensure response is never blank or whitespace
        if not final_response_text or not final_response_text.strip():
            logger.warning("Empty response text after knowledge handling; applying safe fallback.")
            final_response_text = (
                "I'm here to help with your property search. Which project or details would you like to know about?"
            )
            decision.user_facing_response = final_response_text

        # A booking refusal must override the LLM before any tool can run.
        # A CRM follow-up note is not a booking, and this engine has no tool to
        # cancel/remove that note; never claim that it was cancelled.
        if _is_explicit_booking_refusal(cleaned_msg):
            if action_proposed and decision.proposed_action:
                logger.info(
                    "Suppressing proposed action %s after explicit booking refusal: %s",
                    decision.proposed_action.tool_name,
                    cleaned_msg,
                )
            action_proposed = False
            decision.action_proposed = False
            decision.proposed_action = None

            calendar_pending = (
                prior_pending_action in ("create_calendar_event", "find_available_slots")
                or state.pending_action in ("create_calendar_event", "find_available_slots")
            )
            if calendar_pending:
                state.set_pending_action(None)
                state.update_slot("visit_confirmation_pending", None)
                state.update_slot("visit_confirmed", False)
                state.update_slot("available_slots", [])

            previous_agent_lower = (previous_agent_text or "").lower()
            followup_was_recorded = (
                "follow-up request was recorded" in previous_agent_lower
                or "follow up request was recorded" in previous_agent_lower
                or "follow-up request has been recorded" in previous_agent_lower
                or "follow up request has been recorded" in previous_agent_lower
            )
            if followup_was_recorded:
                final_response_text = (
                    "Understood. No site visit has been booked, and no callback was scheduled. "
                    "Your earlier message was a follow-up request, not a booking. I can't cancel or remove a follow-up record through this system."
                )
            else:
                final_response_text = (
                    "Understood. I won't book a site visit. No booking has been made."
                )
            decision.user_facing_response = final_response_text
            decision.detected_intent = "decline_booking"
            decision.knowledge_required = False
            decision.knowledge_query = None

        # 11. Handle External Action Execution & Verification (Phase 5)
        tool_exec_result: Optional[ToolExecutionResult] = None
        if action_proposed and decision.proposed_action and not self.tool_provider:
            action_name = decision.proposed_action.tool_name
            action_arguments = dict(decision.proposed_action.arguments)
            action_proposed = False
            decision.action_proposed = False
            decision.proposed_action = None
            state.set_pending_action(None)
            if action_name == "update_lead" and action_arguments.get("status") == "follow_up_requested":
                final_response_text = (
                    "I can't record your follow-up request because the lead system is unavailable. "
                    "No callback has been scheduled or confirmed."
                )
            else:
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
            # The active conversation state is authoritative.
            # Never trust a domain supplied by the LLM.
            tool_args["domain"] = state.current_domain.value

            # Sensitive authorization must come from trusted session state,
            # never from LLM-generated tool arguments.
            caller_verified_sensitive = (
                state.get_slot("returning_caller_verified_sensitive") is True
                or getattr(state.caller, "verified_sensitive", False) is True
                or state.get_slot("caller_verified_sensitive") is True
            )
            tool_args["caller_verified_sensitive"] = caller_verified_sensitive
            execute_action = True
            allowed_tool_names = {
                "find_available_slots",
                "create_calendar_event",
                "cancel_calendar_event",
                "calendar_cancel_event",
                "update_lead",
                "send_message",
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

            # Sensitive Action Authorization Gate (Phase 3E)
            sensitive_read_tools = {"mail_read_recent", "calendar_get_events"}
            if tool_name in sensitive_read_tools:
                if not state.get_slot("returning_caller_verified_sensitive", False):
                    execute_action = False
                    action_proposed = False
                    decision.action_proposed = False
                    decision.proposed_action = None
                    state.set_pending_action(None)
                    final_response_text = (
                        "I cannot access private records without verified caller authentication. "
                        "Please speak with a customer service representative to verify your identity."
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
                pref_date = (
                    (state.get_slot("preferred_date") if visit_date else None)
                    or tool_args.get("preferred_date")
                    or tool_args.get("date")
                    or state.get_slot("preferred_date")
                    or state.get_slot("meeting_preference")
                )
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
                available_slots = state.get_slot("available_slots") or []
                pending_visit_slot = state.get_slot("visit_confirmation_pending")

                # Deterministic confirmation fix:
                # once we have asked the caller to book a specific offered slot,
                # an explicit "yes" / "book it" must confirm THAT stored slot.
                # Do not let a newly generated LLM slot string or stale
                # meeting_preference cause the confirmation to be asked again.
                if (
                    is_explicit_confirmation
                    and prior_pending_action == "create_calendar_event"
                    and pending_visit_slot
                ):
                    pending_key = re.sub(r"\s+", " ", str(pending_visit_slot)).strip().casefold()
                    exact_pending_slot = next(
                        (
                            str(candidate).strip()
                            for candidate in available_slots
                            if re.sub(r"\s+", " ", str(candidate)).strip().casefold() == pending_key
                        ),
                        None,
                    )
                    if exact_pending_slot:
                        slot = exact_pending_slot
                        tool_args["slot"] = exact_pending_slot
                        tool_args["confirmed"] = True
                        # Keep state and action payload on the same canonical slot.
                        state.update_slot("visit_confirmation_pending", exact_pending_slot)
                        state.update_slot("meeting_preference", exact_pending_slot, sync_caller=True)
                    else:
                        slot = (
                            tool_args.get("slot")
                            or tool_args.get("start_time")
                            or tool_args.get("time")
                            or pending_visit_slot
                            or state.get_slot("meeting_preference")
                        )
                else:
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
                slot_is_available = any(
                    re.sub(r"\s+", " ", str(candidate)).strip().casefold()
                    == re.sub(r"\s+", " ", str(slot)).strip().casefold()
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
                    if "phone_number" not in tool_args:
                        tool_args["phone_number"] = state.caller.phone or state.get_slot("phone")
                    if "attendee_name" not in tool_args:
                        tool_args["attendee_name"] = state.caller.name or state.get_slot("name")
                    if "attendee_email" not in tool_args:
                        tool_args["attendee_email"] = state.caller.email or state.get_slot("email")

            # Action 3b: cancel_calendar_event
            elif tool_name in ("cancel_calendar_event", "calendar_cancel_event"):
                # Safety check: Cancellation can ONLY execute if caller explicitly confirmed
                # an identified appointment in the engine's confirmation flow.
                is_valid_confirmed_flow = (
                    tool_args.get("confirmed") is True
                    and prior_pending_action in ("cancel_calendar_event", "calendar_cancel_event")
                    and bool(pending_cancellation_id)
                    and (is_cancellation_confirmed or is_explicit_confirmation)
                )
                if not is_valid_confirmed_flow:
                    execute_action = False
                    action_proposed = False
                    decision.action_proposed = False
                    decision.proposed_action = None
                    state.set_pending_action(None)
                    final_response_text = (
                        "For safety, appointments cannot be cancelled without your explicit confirmation. "
                        "Would you like me to look up your scheduled visit first?"
                    )
                    decision.user_facing_response = final_response_text
                else:
                    if "phone_number" not in tool_args:
                        tool_args["phone_number"] = state.caller.phone or state.get_slot("phone")
                    caller_verified = (
                        state.get_slot("returning_caller_verified_sensitive", False)
                        or getattr(state.caller, "verified_sensitive", False) is True
                        or state.get_slot("caller_verified_sensitive", False)
                    )
                    tool_args["caller_verified_sensitive"] = caller_verified
                    tool_args["returning_caller"] = state.get_slot("returning_caller", False)
                    if not caller_verified:
                        execute_action = False
                        action_proposed = False
                        decision.action_proposed = False
                        decision.proposed_action = None
                        state.set_pending_action(None)
                        final_response_text = (
                            "For security, cancelling or deleting an appointment requires verified caller authentication. "
                            "Please contact customer support to verify your identity before cancelling."
                        )
                        decision.user_facing_response = final_response_text

            # Action 4: update_lead (CRM lead updates and consultation requests)
            elif tool_name == "update_lead":
                # Protect registered profile details from being overwritten by an unverified returning caller
                if state.get_slot("returning_caller") is True and not state.get_slot("returning_caller_verified_sensitive", False):
                    if tool_args.get("sensitive_action") is True or tool_args.get("overwrite_registered_profile") is True:
                        execute_action = False
                        action_proposed = False
                        decision.action_proposed = False
                        decision.proposed_action = None
                        state.set_pending_action(None)
                        final_response_text = (
                            "For security, modifying registered account details requires verified caller authentication. "
                            "Please contact customer support to verify your identity."
                        )
                        decision.user_facing_response = final_response_text
                    else:
                        # For regular follow-up requests, preserve the trusted registered caller profile
                        if state.caller.name:
                            tool_args["name"] = state.caller.name
                        if state.caller.phone:
                            tool_args["phone"] = state.caller.phone
                        if state.caller.email:
                            tool_args["email"] = state.caller.email

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

                    # Consent applies to the current WhatsApp send only.
                    # Do not carry it forward to a future message.
                    if tool_name == "send_message":
                        state.update_slot("whatsapp_consent", False, sync_caller=True)
                        state.set_pending_action(None)

                    # A successful site-visit booking completes the pending workflow.
                    # Clear the confirmation and offered-slot state so a later "yes"
                    # cannot accidentally trigger the same booking again.
                    if tool_name == "create_calendar_event":
                        state.set_pending_action(None)
                        state.update_slot("visit_confirmation_pending", None)
                        state.update_slot("available_slots", [])
                        state.update_slot("visit_confirmed", True)

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
                            normalized_slots = [str(item) for item in slots]
                            state.update_slot("available_slots", normalized_slots)
                            requested_time = state.get_slot("preferred_time")
                            requested_slot = next(
                                (item for item in normalized_slots if _slot_matches_requested_time(item, requested_time)),
                                None,
                            ) if requested_time else None
                            if requested_slot:
                                state.update_slot("meeting_preference", requested_slot, sync_caller=True)
                                state.update_slot("visit_confirmation_pending", requested_slot)
                                state.update_slot("visit_confirmed", False)
                                speech_confirmation = f"For {proj}, {requested_slot} is available. Would you like me to book that site visit?"
                                state.set_pending_action("create_calendar_event")
                                state.set_pending_question(f"Would you like me to book the site visit for {requested_slot}?")
                            else:
                                slots_str = ", ".join(normalized_slots[:3])
                                if requested_time:
                                    speech_confirmation = f"I couldn't find an available site-visit slot at {requested_time} on {date_str}. For {proj}, these times are available: {slots_str}. Which would you prefer?"
                                else:
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
                    elif tool_name in ("cancel_calendar_event", "calendar_cancel_event"):
                        state.clear_slot("cancellation_confirmation_pending_id")
                        state.clear_slot("cancellation_confirmation_pending_details")
                        speech_confirmation = (
                            "Your appointment has been successfully cancelled. "
                            "The database record and calendar slot have been removed. Is there anything else I can assist you with?"
                        )
                    elif tool_name == "update_lead" and tool_args.get("status") == "follow_up_requested":
                        speech_confirmation = (
                            "The lead system reported that your follow-up request was recorded. "
                            "This system has not scheduled a callback."
                        )
                    else:
                        speech_confirmation = "I have completed that request for you. Is there anything else I can assist you with?"

                    final_response_text = speech_confirmation.strip()
                    decision.user_facing_response = final_response_text
                else:
                    # Execution or verification failed. Do not retain a failed action as pending.
                    state.set_pending_action(None)
                    state.last_tool_result = tool_exec_result
                    action_proposed = False
                    decision.action_proposed = False
                    decision.proposed_action = None
                    if tool_name == "send_message":
                        final_response_text = (
                            "I couldn't send the WhatsApp message because of a system issue. Nothing was sent. "
                            "Would you like to try again or record a follow-up request?"
                        )
                    elif tool_name == "create_calendar_event":
                        final_response_text = (
                            "I couldn't book that site-visit slot. No booking was confirmed. "
                            "Would you like to check other available times?"
                        )
                    elif tool_name in ("cancel_calendar_event", "calendar_cancel_event"):
                        state.update_slot("cancellation_confirmation_pending_id", None)
                        state.update_slot("cancellation_confirmation_pending_details", None)
                        final_response_text = (
                            "I couldn't cancel that appointment due to a system issue. "
                            "The existing appointment was retained in our records. Would you like to try again?"
                        )
                    elif tool_name == "find_available_slots":
                        final_response_text = (
                            "I was unable to retrieve available site visit slots right now. "
                            "Would you like to try another date?"
                        )
                    elif tool_name == "update_lead" and tool_args.get("status") == "follow_up_requested":
                        final_response_text = (
                            "I couldn't record your follow-up request because the lead system returned an error. "
                            "No callback has been scheduled or confirmed."
                        )
                    else:
                        final_response_text = (
                            "I couldn't complete that request because of a system issue. "
                            "No successful action has been confirmed."
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
        if decision.needs_clarification:
            raw_question = (
                decision.clarification_question
                if (decision.clarification_question and str(decision.clarification_question).strip())
                else decision.user_facing_response
            )
            selected_question = str(raw_question).strip() if raw_question else ""
            if selected_question:
                state.set_pending_question(selected_question)
            elif not state.pending_question:
                state.set_pending_question(None)
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
            mask_sensitive_text(cleaned_msg),
            state.metadata.direction.value,
            state.current_domain.value,
            decision.detected_intent,
            state.stage.value,
            len(system_instruction),
            len(user_prompt),
            self.llm.__class__.__name__,
            mask_sensitive_text(decision.user_facing_response if raw_llm_text is None else raw_llm_text),
            decision.detected_domain.value,
            decision.detected_intent,
            decision.proposed_stage.value,
            decision.knowledge_required,
            decision.action_proposed,
            mask_sensitive_text(decision.model_dump_json(exclude={"user_facing_response"})),
            decision.knowledge_required,
            decision.knowledge_query,
            len(grounded_citations),
            grounded_citations,
            decision.proposed_action.tool_name if (action_proposed and decision.proposed_action) else "None",
            mask_sensitive_text(final_response_text),
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
