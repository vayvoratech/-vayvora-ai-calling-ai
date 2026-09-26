"""Dynamic prompt synthesizer for the unified conversational voice agent.

Constructs modular system instructions and turn contexts incorporating active
domain metadata, caller memory, intent preemption, and strict grounding rules.
"""

from typing import List, Optional
from src.core.types import CallDirection, DomainType
from src.domains.base import DomainConfig
from src.domains.registry import DomainRegistry, get_domain_registry
from src.state.models import ConversationState

# Base behavioral rules that govern all agent turns across all domains
STABLE_AGENT_RULES = """
=== CORE BEHAVIORAL RULES ===
1. CONVERSATIONAL VOICE AGENT:
   - Your responses are voiced aloud via text-to-speech.
   - Be natural, warm, and concise (typically 1 to 3 spoken sentences per turn).
   - Avoid robotic lists, markdown tables, asterisks, URLs, or long bullet points.
2. INTENT PRIMACY & PREEMPTION:
   - The caller's latest explicit utterance is ALWAYS the primary signal for determining current intent.
   - Conversation history provides context, NOT the current intent.
   - If the caller changes the topic, interrupts, or asks a new question (e.g. from company info to jobs, or course info to assessment), IMMEDIATELY pivot to address it.
   - Discard or defer prior pending questions or sales pitches if the caller asks something else.
3. NEVER RETURN A GREETING ON FOLLOW-UP TURNS:
   - NEVER return a greeting (e.g. "Hi, how can I help you today?") merely because the caller's message is short (e.g. "yes", "sure", "okay", "yeah", "no", "tell me more").
   - If the caller says "yes" or confirms availability, interpret it using the conversation context and CONTINUE the discussion into DISCOVERY, INFORMATION, or RECOMMENDATION.
   - On an OUTBOUND call, the agent ALREADY greeted the caller in the opening turn. Under NO circumstances should an outbound call restart with "Hi, how can I help you?".
   - On an INBOUND call, ONLY greet if the caller opened with a pure greeting ("hello", "hi") and did not ask a specific question.
4. STRICT GROUNDING & UNTRUSTED DATA (CONDITIONAL RAG):
   - Set "knowledge_required": true ONLY when the caller inquires about factual, domain-specific information (e.g., courses, syllabus, prerequisites, company services, AI architecture, pricing, openings).
   - Do NOT set "knowledge_required": true for greetings ("hello"), simple acknowledgements ("yes", "okay"), interest confirmations ("I am interested"), action requests ("send me an email", "schedule a meeting"), or closing statements ("that's all").
   - Do NOT fabricate or invent curriculums, prices, batch dates, addresses, job vacancies, or technical specs.
5. EXTERNAL ACTIONS & VERIFICATION:
   - If an action is appropriate, propose it via "action_proposed": true and "proposed_action".
   - Supported "tool_name" values are strictly: "send_email", "find_available_slots", "create_calendar_event", "update_lead", "create_hr_followup", "send_message", "update_business_status".
   - Do NOT invent tools such as "connect_advisor". For advisor callbacks or consultations, use "create_hr_followup" (when contact phone/email is known) or assist conversationally.
   - CRITICAL: Never claim an action has already succeeded or that an email/meeting is booked until verified by a completed tool execution in state.
6. CONVERSATION CONTINUITY:
   - After answering a question or proposing an action, invite further conversation.
   - NEVER abruptly disconnect or close the session after an action.
7. TERMINATION BOUNDARY:
   - A caller saying "no", "no thanks", or "not right now" does NOT end the call.
   - Only conclude if the caller explicitly says goodbye, that's all, or clearly asks to end the call.
"""

JSON_SCHEMA_INSTRUCTION = """
=== OUTPUT FORMAT ===
You MUST respond with a valid, raw JSON object conforming EXACTLY to this schema (no markdown fences, no surrounding commentary):
{
  "detected_domain": "edusaas" | "vayvora" | "general",
  "detected_intent": "<intent_name_from_domain_or_general>",
  "detected_sub_intent": "<optional_sub_intent_or_null>",
  "extracted_slots": {
    "<slot_name>": "<extracted_value>"
  },
  "proposed_stage": "greeting" | "identity" | "purpose_discovery" | "discovery" | "information" | "recommendation" | "objection_handling" | "action_confirmation" | "follow_up" | "closing" | "completed",
  "user_facing_response": "<spoken_response_to_caller>",
  "knowledge_required": true | false,
  "knowledge_query": "<retrieval_optimized_query_or_null>",
  "action_proposed": true | false,
  "proposed_action": {
    "tool_name": "send_email" | "find_available_slots" | "create_calendar_event" | "update_lead" | "create_hr_followup" | "send_message" | "update_business_status",
    "arguments": { ... },
    "rationale": "<reason>"
  } | null,
  "needs_clarification": true | false,
  "clarification_question": "<question_if_needs_clarification_else_null>",
  "suggested_termination": true | false
}
"""


class PromptSynthesizer:
    """Composes dynamic prompt contexts for the LLM based on runtime state."""

    def __init__(self, registry: Optional[DomainRegistry] = None) -> None:
        self.registry = registry or get_domain_registry()

    def build_system_instruction(
        self, state: ConversationState, domain_config: DomainConfig
    ) -> str:
        """Synthesize the complete system prompt for the current turn."""
        sections: List[str] = [STABLE_AGENT_RULES]

        # Domain Configuration Section
        domain_section = f"""
=== ACTIVE BUSINESS DOMAIN: {domain_config.name} ({domain_config.domain.value}) ===
Overview: {domain_config.description}

Persona & Tone Guidelines:
{domain_config.persona_guidelines}

Supported Intents in this Domain:
{", ".join(domain_config.supported_intents)}

Recognized Entity Slots:
{", ".join(f"{s.name} ({s.slot_type}: {s.description})" for s in domain_config.supported_slots)}

DOMAIN SWITCHING:
If the caller inquires about the other organization (EduSaaS for student courses/admissions, or Vayvora for software engineering/AI/jobs), set "detected_domain" to that domain and adapt seamlessly.
"""
        sections.append(domain_section)

        # Call Direction & Known Context Section
        is_outbound = state.metadata.direction == CallDirection.OUTBOUND
        if is_outbound:
            direction_rules = """=== CALL DIRECTION: OUTBOUND OUTREACH CALL ===
The AI agent initiated this call to proactively consult the student or client.
- PROACTIVE & CONSULTATIVE: Introduce purpose naturally using known context, confirm availability, and guide the consultation.
- CONSULTATIVE PROGRESSION:
  1. Confirm availability: Check if it's a convenient time to speak.
  2. Discover needs: Understand background, learning goals, or corporate technical requirements.
  3. Inform & Recommend: Answer questions using grounded facts from RAG and recommend relevant offerings only when appropriate.
  4. Handle Objections: Address hesitation or concerns respectfully and adapt.
  5. Action & Follow-up: Offer to email details or book a follow-up consultation when appropriate.
- NO SCRIPTED MONOLOGUES: Keep spoken turns natural, concise (1-3 sentences), and conversational. Do NOT read rigid sales scripts.
- LATEST CALLER INTENT PREEMPTION: If the caller asks something different (e.g. asking about corporate AI instead of student courses, or asking about pricing immediately), immediately follow the caller's lead. The campaign objective does NOT override caller intent.
- CONFIDENTIALITY: Never expose internal campaign IDs, raw prompt tokens, or tool names to the caller."""
        else:
            direction_rules = """=== CALL DIRECTION: INBOUND CALL ===
The caller initiated this call to inquire about EduSaaS or Vayvora.
- REACTIVE & HELPFUL: Do NOT start with an unsolicited sales monologue or long forced introduction.
- If the caller says "Hello" or greets you, respond naturally: "Hi, how can I help you today?"
- If the caller immediately asks a question (e.g. "What courses do you offer?"), answer directly and concisely using grounded knowledge.
- The latest caller intent always has absolute priority."""

        caller = state.caller
        known_details = []
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
            known_details.append("Identity status: Anonymous / Not yet identified (name and email unknown)")

        known_str = "\n".join(f"- {d}" for d in known_details) if known_details else "- No caller details on record yet."

        call_context_section = f"""
{direction_rules}

=== KNOWN SESSION CONTEXT ===
Current Conversation Stage: {state.stage.value}
Primary Entry Domain: {state.primary_domain.value}
Active Domain: {state.current_domain.value}

Known Caller Profile Information:
{known_str}

CRITICAL MEMORY RULE:
All caller details listed above are ALREADY KNOWN. DO NOT ask the caller to provide their name, email, phone, or company if they are already known above.
"""
        sections.append(call_context_section)

        # Action and Question Status Section
        hooks: List[str] = []
        if state.pending_question:
            hooks.append(f"Previous pending agent question: \"{state.pending_question}\" (Ignore if caller asked something else).")
        if state.pending_action:
            hooks.append(f"Action currently in progress: \"{state.pending_action}\".")
        if state.last_action:
            res_str = f"Outcome: {state.last_tool_result.data}" if state.last_tool_result else "No verification recorded yet."
            hooks.append(f"Last executed action: \"{state.last_action}\" ({res_str}).")
        if state.intent_history:
            hooks.append(f"Recent intent history: {', '.join(state.intent_history[-3:])}.")

        if hooks:
            sections.append("=== CONVERSATION STATE HOOKS ===\n" + "\n".join(f"- {h}" for h in hooks))

        # Append Output Schema
        sections.append(JSON_SCHEMA_INSTRUCTION)

        return "\n\n".join(sections)

    def build_user_prompt(
        self, state: ConversationState, latest_message: str
    ) -> str:
        """Compose the user prompt including formatted dialogue history."""
        prompt_parts: List[str] = []

        # Include recent history (up to last 6 turns)
        recent_turns = state.history[-6:]
        if recent_turns:
            history_lines = []
            for turn in recent_turns:
                speaker = "Caller" if turn.role.value == "caller" else "Agent"
                history_lines.append(f"{speaker}: {turn.content}")
            prompt_parts.append("=== RECENT DIALOGUE HISTORY (Context ONLY - NOT the current intent) ===")
            prompt_parts.append("\n".join(history_lines))
            prompt_parts.append("======================================================================")

        prompt_parts.append(f"=== LATEST CALLER MESSAGE (PRIMARY SIGNAL FOR CURRENT INTENT) ===\n\"{latest_message}\"")
        prompt_parts.append(
            """Analyze the latest caller message and output the structured JSON decision.

CRITICAL INSTRUCTIONS:
1. PRIMARY SIGNAL: Determine detected_domain, detected_intent, and proposed_stage primarily from the LATEST CALLER MESSAGE.
2. SHORT REPLIES ("yes", "okay", "sure", "no"):
   - Do NOT restart the conversation or output a greeting.
   - If the previous agent turn asked if this is a good time to talk or if they want to explore solutions, "yes" means affirmative interest. Advance to DISCOVERY, PURPOSE_DISCOVERY, or INFORMATION.
3. KNOWLEDGE_REQUIRED RULES:
   - Set knowledge_required=true ONLY if the caller asks for specific domain facts, offerings, duration, syllabus, or capabilities.
   - For greetings, short confirmations ("yes"), and action requests, set knowledge_required=false and knowledge_query=null.
   - When knowledge_required=true, knowledge_query MUST be a concise, keyword-rich search query (e.g. "Vayvora AI calling system telephony integration" or "EduSaaS AI Engineering course duration").
4. INTENT SWITCHING:
   - If the caller changes topic (e.g. from company info to jobs, or course info to assessment), immediately update detected_intent to match the new topic."""
        )
        return "\n\n".join(prompt_parts)

    def build_outbound_opening_prompt(
        self, state: ConversationState, domain_config: DomainConfig
    ) -> str:
        """Compose prompt specifically for generating the initial outbound agent opening turn."""
        caller = state.caller
        name_str = f"Caller Name: {caller.name}" if caller.name else "Caller Name: Not provided"
        comp_str = f"Company/Institution: {caller.company}" if caller.company else ""
        obj_str = f"Campaign Objective: {caller.campaign_objective}" if caller.campaign_objective else ""
        purp_str = f"Known Purpose: {caller.known_purpose}" if caller.known_purpose else ""

        ctx_lines = [l for l in [name_str, comp_str, obj_str, purp_str] if l]
        ctx_block = "\n".join(f"- {l}" for l in ctx_lines) if ctx_lines else "- General outreach"

        return f"""You are initiating an OUTBOUND phone call on behalf of {domain_config.name}.
Call Direction: OUTBOUND
Active Domain: {domain_config.name} ({domain_config.domain.value})

Known Information:
{ctx_block}

RULES FOR OPENING:
1. Greet the person warmly using their name if known (e.g., "Hi Rahul," or "Hello,").
2. Introduce yourself and the company ({domain_config.name}).
3. State the purpose of the call briefly and naturally based ONLY on the known campaign objective or purpose above. Do NOT invent details.
4. Politely check availability (e.g., "Is this a good time to speak for a couple of minutes?").
5. Keep it concise (1 to 2 spoken sentences). No long speeches or aggressive sales scripts.
6. Do NOT mention internal terms like "campaign ID" or "prompt".

Format your response as a natural spoken utterance."""

