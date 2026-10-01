"""Health check endpoints for microservice monitoring."""

from __future__ import annotations

import asyncio
import time
from fastapi import APIRouter, Depends
from src.api.dependencies import get_workbench_service
from src.api.schemas import HealthResponse
from src.ui.service import WorkbenchService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check(
    service: WorkbenchService = Depends(get_workbench_service),
) -> HealthResponse:
    """Return health status across all agent subsystems."""
    redis_healthy = True
    store = getattr(service.knowledge_provider, "vector_store", None) or getattr(service.knowledge_provider, "store", None)
    if store and hasattr(store, "health_check"):
        try:
            redis_healthy = await asyncio.wait_for(store.health_check(), timeout=1.0)
        except Exception:
            redis_healthy = False

    runtime_mode = getattr(service, "overall_mode", "LIVE")
    if hasattr(runtime_mode, "value"):
        runtime_mode = runtime_mode.value

    postgres_healthy = False
    try:
        from src.database.connection import check_postgres_health
        postgres_healthy = await asyncio.wait_for(check_postgres_health(), timeout=1.5)
    except Exception:
        postgres_healthy = False

    subsystems = {
        "runtime_mode": str(runtime_mode),
        "llm_provider": type(service.llm_provider).__name__,
        "knowledge_provider": type(service.knowledge_provider).__name__,
        "redis_connected": redis_healthy,
        "postgres_connected": postgres_healthy,
        "tool_provider": type(service.tool_provider).__name__,
        "vad_provider": type(service.vad_provider).__name__,
        "stt_provider": type(service.stt_provider).__name__,
        "tts_provider": type(service.tts_provider).__name__,
    }

    components = {
        "engine": "ConversationEngine",
        "state_manager": "ConversationStateManager",
        "redis": redis_healthy,
        "postgres": postgres_healthy,
        "llm": type(service.llm_provider).__name__,
        "rag": type(service.knowledge_provider).__name__,
        "vad": type(service.vad_provider).__name__,
        "stt": type(service.stt_provider).__name__,
        "tts": type(service.tts_provider).__name__,
    }

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=time.time(),
        subsystems=subsystems,
        components=components,
    )
