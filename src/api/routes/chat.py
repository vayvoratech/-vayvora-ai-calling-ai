"""REST Chat and Configuration Endpoints."""

from __future__ import annotations

import time
from fastapi import APIRouter, Depends, HTTPException
from src.api.dependencies import get_conversation_engine, get_state_manager
from src.api.schemas import ChatRequest, ChatResponse, ConfigResponse
from src.config import get_settings
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.types import CallDirection, DomainType
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState

router = APIRouter(tags=["REST Communication"])


@router.get("/config", response_model=ConfigResponse)
@router.get("/api/v1/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Return central configuration parameters."""
    cfg = get_settings()
    return ConfigResponse(
        model_name=cfg.gemini_model,
        llm_provider=cfg.llm_provider,
        default_domain="edusaas",
        relevance_threshold=cfg.rag_relevance_threshold,
        stt_provider=cfg.stt_provider,
        tts_provider=cfg.tts_provider,
        vad_provider=cfg.vad_provider,
    )


@router.post("/chat", response_model=ChatResponse)
@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    payload: ChatRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
    state_manager: ConversationStateManager = Depends(get_state_manager),
) -> ChatResponse:
    """Process a single conversational turn via REST and return structured decision telemetry."""
    session_id = payload.session_id or "default_session"
    raw_domain = (payload.domain or "edusaas").lower()
    domain = DomainType.VAYVORA if raw_domain == "vayvora" else DomainType.EDUSAAS

    # Retrieve or create session state
    state = state_manager.get(session_id)
    if state is None:
        state = state_manager.create_session(
            session_id=session_id,
            domain=domain,
            direction=CallDirection.INBOUND,
            caller_profile=CallerProfile(phone=None, name=None, email=None),
        )

    # Inbound identity guardrail: do NOT inject sample/mock caller data into slots
    for k, v in payload.slots.items():
        if v and str(v).lower() not in ["default caller", "caller@mail.com", "unknown@example.com"]:
            state.extracted_slots[k] = v

    t0 = time.perf_counter()
    try:
        turn_result: EngineTurnResult = await engine.process_user_turn(
            state=state,
            user_message=payload.query_text,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        tool_input = None
        action_rationale = ""
        if turn_result.decision.action_proposed and turn_result.decision.proposed_action:
            tool_input = turn_result.decision.proposed_action.arguments
            action_rationale = turn_result.decision.proposed_action.rationale or ""

        return ChatResponse(
            response=turn_result.response_text or "No response generated.",
            latency_ms=round(elapsed_ms, 2),
            route=turn_result.decision.detected_intent or "general",
            confidence=1.0,
            margin=0.0,
            rationale=action_rationale,
            tool_name=turn_result.tool_result.tool_name if turn_result.tool_result else None,
            tool_input=tool_input,
            tool_result=turn_result.tool_result.data if turn_result.tool_result else None,
            slots=dict(state.extracted_slots),
            grounded_citations=turn_result.grounded_citations,
            stage=state.stage.value,
            conversation_active=state.conversation_active,
            error=None,
            session_id=session_id,
            domain=domain.value,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return ChatResponse(
            response="I apologize, but an error occurred while processing your request.",
            latency_ms=round(elapsed_ms, 2),
            route="error",
            confidence=0.0,
            margin=0.0,
            rationale=str(exc),
            slots=dict(state.extracted_slots),
            stage=state.stage.value,
            conversation_active=state.conversation_active,
            error=str(exc),
        )
