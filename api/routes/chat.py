import time
from fastapi import APIRouter
from api.dependencies import get_agent_runtime
from api.schemas import ChatRequest, ChatResponse, ConfigResponse
from src.agents.config import DEFAULT_AGENT_CONFIG
router = APIRouter(prefix="/api/v1", tags=["REST Communication"])


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    return ConfigResponse(
        model_name=DEFAULT_AGENT_CONFIG.model_name,
        router_confidence_threshold=DEFAULT_AGENT_CONFIG.router_confidence_threshold,
        router_temperature=DEFAULT_AGENT_CONFIG.router_temperature,
        kb_resonance_weight=DEFAULT_AGENT_CONFIG.kb_resonance_weight,
        tool_similarity_threshold=DEFAULT_AGENT_CONFIG.tool_similarity_threshold,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    agent = get_agent_runtime()

    formatted_messages = [
        {"role": m.role, "content": m.content} for m in payload.messages
    ]

    state = {
        "session_id": payload.session_id,
        "user_input": payload.user_input,
        "tenant_id": payload.tenant_id,
        "messages": formatted_messages,
        "slots": payload.slots,
    }

    t0 = time.perf_counter()
    result = await agent.run(state)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ChatResponse(
        response=result.get("response", "No response generated."),
        latency_ms=round(elapsed_ms, 2),
        route=result.get("route", "unknown"),
        confidence=result.get("route_confidence", 0.0),
        margin=result.get("route_margin", 0.0),
        rationale=result.get("route_rationale", ""),
        tool_name=result.get("tool_name"),
        tool_input=result.get("tool_input"),
        tool_result=result.get("tool_result"),
        slots=result.get("slots", {}),
        error=result.get("error"),
    )