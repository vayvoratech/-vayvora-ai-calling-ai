"""Conversation engine orchestrating prompt synthesis, LLM generation, and state updates.

Coordinates turn-by-turn conversational flow, ensuring intent preemption,
non-fabrication of tool execution, and explicit termination enforcement.
"""

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.core.decision import ConversationalDecision, DecisionValidator, ProposedAction
from src.core.interfaces import KnowledgeProvider, LLMProvider, ToolProvider
from src.core.prompts import PromptSynthesizer
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    RAGQuery,
    ToolCallRequest,
    ToolExecutionResult,
    TurnRole,
)
from src.domains.base import DomainConfig
from src.domains.registry import DomainRegistry, get_domain_registry
from src.logging import get_logger
from src.state.models import ConversationState

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
        self.registry = registry or get_domain_registry()
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
        system_instruction = (
            f"You are the professional voice agent for {domain_config.name} initiating an outbound outreach call. "
            "Speak naturally, concisely, and warmly. Introduce yourself, state the reason for calling based on known context, "
            "and check availability in 1-2 spoken sentences."
        )

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
        if (
            not opening_text
            or len(opening_text.strip()) < 10
            or opening_text.strip().startswith("{")
            or opening_text.strip() == "Mock default response."
        ):
            opening_text = self._generate_deterministic_outbound_opening(state, domain_config)

        # 4. Construct initial ConversationalDecision
        decision = ConversationalDecision(
            detected_domain=state.current_domain,
            detected_intent="outbound_greeting",
            proposed_stage=ConversationStage.GREETING,
            user_facing_response=opening_text,
            action_proposed=False,
            knowledge_required=False,
        )

        # 5. Update state
        state.set_stage(ConversationStage.GREETING)
        state.set_intent("outbound_greeting", clear_pending_question=False)
        pending_q = (
            "Is this a good time for a quick conversation?"
            if state.current_domain == DomainType.VAYVORA
            else "Is this a good time to speak for a couple of minutes?"
        )
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

    def _generate_deterministic_outbound_opening(
        self, state: ConversationState, domain_config: DomainConfig
    ) -> str:
        """Produce a natural, professional opening utterance using known context without fabricating details."""
        caller = state.caller
        name = caller.name.strip() if caller.name else None
        company = caller.company.strip() if caller.company else None
        obj = caller.campaign_objective.strip() if caller.campaign_objective else None
        purpose = caller.known_purpose.strip() if caller.known_purpose else None

        reason = obj or purpose

        if state.current_domain == DomainType.EDUSAAS:
            greeting_name = f"Hi {name}, " if name else "Hello, "
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
                return f"{greeting_name}this is the EduSaaS team. I'm reaching out because you had shown interest in {r_clean}. Is this a good time to speak for a couple of minutes?"
            return f"{greeting_name}this is the EduSaaS team. I'm reaching out because you had shown interest in our courses. Is this a good time to speak for a couple of minutes?"

        else:  # Vayvora
            greeting_name = f"Hi {name}, " if name else "Hello, "
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
                return f"{greeting_name}this is the Vayvora team. I'm reaching out to understand whether {target_org} is currently exploring {r_clean}. Is this a good time for a quick conversation?"
            return f"{greeting_name}this is the Vayvora team. I'm reaching out to understand whether {target_org} is currently exploring AI or automation solutions. Is this a good time for a quick conversation?"

    async def process_user_turn(
        self, state: ConversationState, user_message: str
    ) -> EngineTurnResult:
        """Process an incoming caller utterance and advance conversation state."""
        cleaned_msg = user_message.strip()

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

        # 2. Retrieve active domain configuration
        domain_config = self.registry.get_or_raise(state.current_domain)

        # 3. Synthesize prompts
        t_prompt = time.perf_counter()
        system_instruction = self.synthesizer.build_system_instruction(state, domain_config)
        user_prompt = self.synthesizer.build_user_prompt(state, cleaned_msg)
        timing["prompt_construction_ms"] = round((time.perf_counter() - t_prompt) * 1000, 2)

        # 4. Generate structured decision from LLM
        t_llm = time.perf_counter()
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
        timing["llm_decision_ms"] = round((time.perf_counter() - t_llm) * 1000, 2)

        # Inbound simple greeting guard: respond naturally without forced introduction or sales scripts
        if (
            state.metadata.direction == CallDirection.INBOUND
            and cleaned_msg.lower().rstrip(".!?") in ("hello", "hi", "hey", "hello there", "hi there")
            and decision.detected_intent in ("greeting", "general_inquiry", "unknown", None)
            and not decision.knowledge_required
            and not decision.action_proposed
        ):
            decision.user_facing_response = "Hi, how can I help you today?"

        # 5. Domain Switching
        if decision.detected_domain != state.current_domain:
            logger.info(
                "Domain switch triggered: %s -> %s",
                state.current_domain.value,
                decision.detected_domain.value,
            )
            state.switch_domain(decision.detected_domain)

        # 6. Intent Preemption & Updating
        # The latest explicit user intent has priority over prior pending questions
        if decision.detected_intent:
            state.set_intent(
                new_intent=decision.detected_intent,
                sub_intent=decision.detected_sub_intent,
                clear_pending_question=True,
            )

        # 7. Slot Updates
        for slot_key, slot_val in decision.extracted_slots.items():
            state.update_slot(slot_key, slot_val, sync_caller=True)

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
        if action_proposed and decision.proposed_action:
            state.set_pending_action(decision.proposed_action.tool_name)

        # 10. Handle Grounded Knowledge Retrieval (Phase 4)
        grounded_citations: List[str] = []
        final_response_text = decision.user_facing_response

        if decision.knowledge_required and self.knowledge_provider:
            t_rag = time.perf_counter()
            query_text = decision.knowledge_query or decision.detected_intent or cleaned_msg
            rag_query = RAGQuery(
                domain=decision.detected_domain,
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
                    grounding_prompt = f"""You are formulating a spoken answer for the caller based strictly on verified reference data.
Caller said: "{cleaned_msg}"
Active Domain: {decision.detected_domain.value}

{grounded_res.formatted_context}

RULES:
1. Provide a natural spoken response (1 to 3 sentences) using ONLY facts from the reference data above.
2. If the reference data does not answer the question, state clearly that you do not have that specific detail.
3. NEVER invent unverified facts, prices, policies, or dates.
4. Do NOT mention "reference data" or internal document titles to the caller."""

                    t_ground = time.perf_counter()
                    grounded_speech = await self.llm.generate_response(
                        prompt=grounding_prompt,
                        system_instruction="You are a voice agent responding strictly from verified reference data.",
                    )
                    timing["grounded_llm_ms"] = round((time.perf_counter() - t_ground) * 1000, 2)
                    final_response_text = grounded_speech.strip()
                    decision.user_facing_response = final_response_text
            except Exception as rag_err:
                timing["rag_retrieval_ms"] = round((time.perf_counter() - t_rag) * 1000, 2)
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

            # Auto-populate caller profile details into arguments if absent
            if "caller_name" not in tool_args and state.caller.name:
                tool_args["caller_name"] = state.caller.name
            if "caller_email" not in tool_args and state.caller.email:
                tool_args["caller_email"] = state.caller.email
            if "email" not in tool_args and state.caller.email:
                tool_args["email"] = state.caller.email
            if "recipient" not in tool_args and state.caller.email:
                tool_args["recipient"] = state.caller.email
            if "caller_phone" not in tool_args and state.caller.phone:
                tool_args["caller_phone"] = state.caller.phone
            if "domain" not in tool_args:
                tool_args["domain"] = state.current_domain.value

            # Guard 1: Email Safety - missing or invalid recipient email
            recipient = (tool_args.get("recipient") or tool_args.get("email") or state.caller.email or "").strip()
            if tool_name == "send_email" and (not recipient or "@" not in recipient):
                final_response_text = "I would be happy to email those details to you. Could you please share your email address?"
                decision.user_facing_response = final_response_text
                state.set_pending_question("Could you please share your email address?")
            # Guard 2: Calendar Safety - missing confirmed slot
            elif tool_name == "create_calendar_event":
                meeting_preference = state.get_slot("meeting_preference")

                if meeting_preference and not (
                    tool_args.get("slot")
                    or tool_args.get("start_time")
                    or tool_args.get("time")
                ):
                    tool_args["slot"] = meeting_preference

                if not (
                    tool_args.get("slot")
                    or tool_args.get("start_time")
                    or tool_args.get("time")
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
                # Dispatch to tool provider
                tool_call_req = ToolCallRequest(
                    tool_name=tool_name,
                    arguments=tool_args,
                    call_id=state.metadata.call_id,
                )
                t_tool = time.perf_counter()
                tool_exec_result = await self.tool_provider.execute_tool(tool_call_req)
                timing["tool_execution_ms"] = round((time.perf_counter() - t_tool) * 1000, 2)

                if tool_exec_result.success:
                    # Verified Success: update state
                    state.complete_action(tool_name, tool_exec_result)

                    if tool_name == "update_business_status":
                        new_status = tool_exec_result.data.get("status") or tool_args.get("status")
                        if new_status:
                            state.update_business_status(str(new_status))

                    # Synthesize verbal confirmation incorporating verified reference
                    action_prompt = f"""An action requested by the caller has been EXECUTED and VERIFIED successfully.
Caller said: "{cleaned_msg}"
Action: {tool_name}
Verification Code: {tool_exec_result.verification_code}
Data: {tool_exec_result.data}

Formulate a natural, courteous spoken response (1-2 sentences) confirming the action was completed and inviting any further questions.
Do NOT fabricate delivery details beyond what is confirmed above."""

                    speech_confirmation = await self.llm.generate_response(
                        prompt=action_prompt,
                        system_instruction="You are a voice agent confirming a verified action.",
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
                        if tool_name == "send_email":
                            recipient_addr = recipient or "your email"
                            speech_confirmation = f"I have sent the email with the requested details to {recipient_addr}. Is there anything else I can help you with today?"
                        elif tool_name == "create_calendar_event":
                            slot = tool_args.get("slot", "the requested time")
                            speech_confirmation = f"I have scheduled your consultation for {slot}. Is there anything else I can assist you with?"
                        else:
                            speech_confirmation = "I have completed that request for you. Is there anything else I can assist you with?"

                    final_response_text = speech_confirmation.strip()
                    decision.user_facing_response = final_response_text
                else:
                    # Execution or verification failed
                    state.pending_action = None
                    state.last_tool_result = tool_exec_result
                    err_msg = tool_exec_result.error_message or "action could not be verified"
                    logger.warning("Action %s failed: %s", tool_name, err_msg)
                    final_response_text = (
                        "I attempted to complete that request, but encountered a system issue. "
                        "I have noted this down so our team can follow up with you directly."
                    )
                    decision.user_facing_response = final_response_text
                    state.conversation_active = True

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
