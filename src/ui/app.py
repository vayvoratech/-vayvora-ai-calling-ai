"""FastAPI Testing Workbench UI for the Unified AI Voice Agent.

Provides a lightweight HTML+CSS+JS testing console allowing interactive testing
of the conversation engine, intent routing, grounded RAG, and verified MCP actions.
Replaces the legacy Streamlit UI while preserving the full ConversationEngine architecture.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.core.types import CallDirection, DomainType
from src.logging import get_logger
from src.ui.bootstrap import bootstrap_workbench
from src.ui.service import WorkbenchService

logger = get_logger("ui.app")

# Base directories
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Unified AI Voice Agent Testing Workbench",
    description="Interactive conversational testing console for Inbound and Outbound voice agent flows.",
    version="2.0.0",
)

# Static and Templates mounting
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Lazy-loaded singleton service
_workbench_service: Optional[WorkbenchService] = None


def get_service() -> WorkbenchService:
    """Return initialized WorkbenchService singleton."""
    global _workbench_service
    if _workbench_service is None:
        logger.info("Bootstrapping WorkbenchService singleton...")
        _workbench_service = bootstrap_workbench()
    return _workbench_service


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    direction: str = Field(default="inbound", description="Call direction: 'inbound' or 'outbound'")
    domain: str = Field(default="vayvora", description="Domain: 'vayvora' or 'edusaas'")
    caller_name: str = Field(default="Valued Caller", description="Caller/Customer name")
    caller_phone: str = Field(default="+15551234567", description="Caller phone number")
    caller_email: Optional[str] = Field(default="caller@example.com", description="Caller email")
    caller_company: Optional[str] = Field(default="Acme Corp", description="Caller company")
    campaign_objective: Optional[str] = Field(
        default="Product follow-up consultation",
        description="Outbound campaign objective",
    )
    known_purpose: Optional[str] = Field(
        default="AI consulting services",
        description="Known purpose or interest of the contact",
    )


class TurnRequest(BaseModel):
    user_message: str = Field(..., description="Message spoken or typed by the user/caller")


# -----------------------------------------------------------------------------
# Web & API Endpoints
# -----------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serve the main HTML workbench console."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )

@app.get("/api/status")
async def get_system_status() -> Dict[str, Any]:
    """Retrieve runtime subsystem statuses and initialization notes."""
    service = get_service()
    return {
        "status": "ready",
        "overall_mode": service.overall_mode.value,
        "llm_mode": service.llm_mode,
        "rag_mode": service.rag_mode,
        "tool_mode": service.tool_mode,
        "vad_mode": service.vad_mode,
        "stt_mode": service.stt_mode,
        "tts_mode": service.tts_mode,
        "initialization_notes": service.initialization_notes,
    }


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest) -> Dict[str, Any]:
    """Create a new conversational session (Inbound or Outbound)."""
    service = get_service()
    call_id = f"session-{uuid.uuid4().hex[:8]}"

    # Resolve domain enum
    try:
        domain = DomainType(req.domain.lower())
    except ValueError:
        domain = DomainType.VAYVORA

    # Resolve direction enum
    try:
        direction = CallDirection(req.direction.lower())
    except ValueError:
        direction = CallDirection.INBOUND

    opening_message = None

    if direction == CallDirection.OUTBOUND:
        state = service.create_outbound_session(
            call_id=call_id,
            caller_phone=req.caller_phone,
            domain=domain,
            caller_name=req.caller_name,
            campaign_id="CAMP-OUTBOUND-01",
            campaign_objective=req.campaign_objective or "Product follow-up consultation",
            caller_email=req.caller_email,
            company=req.caller_company,
            known_purpose=req.known_purpose,
        )
        if service.last_turn_result:
            opening_message = service.last_turn_result.response_text
        diagnostics = service.get_debug_payload(state, service.last_turn_result)
    else:
        state = service.create_inbound_session(
            call_id=call_id,
            caller_phone=req.caller_phone,
            domain=domain,
            caller_name=req.caller_name,
            caller_email=req.caller_email,
            caller_company=req.caller_company,
        )
        diagnostics = service.get_debug_payload(state)

    return {
        "session_id": call_id,
        "direction": direction.value,
        "domain": domain.value,
        "opening_message": opening_message,
        "diagnostics": diagnostics,
    }


@app.post("/api/sessions/{session_id}/turns")
async def process_turn(session_id: str, req: TurnRequest) -> Dict[str, Any]:
    """Process a user conversational turn via the unified ConversationEngine."""
    service = get_service()
    state = service.get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    try:
        updated_state, turn_result = await service.process_turn_async(
            session_id=session_id, user_message=req.user_message
        )
    except Exception as exc:
        logger.error("Failed to process turn for session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    diagnostics = service.get_debug_payload(updated_state, turn_result)

    return {
        "session_id": session_id,
        "response_text": turn_result.response_text,
        "latency_ms": turn_result.latency_ms,
        "diagnostics": diagnostics,
    }


@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str) -> Dict[str, Any]:
    """Reset and delete an active session."""
    service = get_service()
    service.reset_session(session_id)
    return {"status": "ok", "session_id": session_id}


def run():
    """CLI entrypoint to run the FastAPI testing workbench."""
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info("Starting Voice Agent Workbench on http://%s:%s", host, port)
    uvicorn.run("src.ui.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run()
