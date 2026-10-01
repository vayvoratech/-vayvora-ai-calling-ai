"""Voice Telephony and Real-Time WebSocket Streaming Router.

Provides:
- TwiML telephony webhook (`POST /voice` and `POST /api/v1/voice`)
- Real-time media streaming WebSocket (`/media-stream`)
- Session start, stop, and status management endpoints
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, WebSocket, WebSocketDisconnect

from src.api.dependencies import get_conversation_engine, get_state_manager, get_voice_session_manager
from src.api.schemas import SessionResponse, SessionStartRequest
from src.core.engine import ConversationEngine
from src.core.types import CallDirection, DomainType
from src.logging import get_logger
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.voice.session_manager import VoiceSessionManager

logger = get_logger("api.routes.voice")

router = APIRouter(tags=["Voice Streaming"])

PUBLIC_WS_URL = os.getenv("PUBLIC_WS_URL", "wss://your-domain.com/media-stream")


@router.post("/voice")
@router.post("/api/v1/voice")
async def telephony_webhook() -> Response:
    """TwiML / Telephony connector endpoint responding with media stream URL."""
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{PUBLIC_WS_URL}" />
    </Connect>
</Response>
"""
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/media-stream")
async def voice_media_stream(
    websocket: WebSocket,
    session_id: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default="edusaas"),
    direction: Optional[str] = Query(default="inbound"),
    caller_phone: Optional[str] = Query(default=None),
    caller_name: Optional[str] = Query(default=None),
    manager: VoiceSessionManager = Depends(get_voice_session_manager),
):
    """Real-time bidirectional media stream WebSocket endpoint."""
    raw_dir = (direction or "inbound").lower()
    call_dir = CallDirection.OUTBOUND if raw_dir == "outbound" else CallDirection.INBOUND

    raw_domain = (domain or "edusaas").lower()
    dom = DomainType.VAYVORA if raw_domain == "vayvora" else DomainType.EDUSAAS
    initial_domain = DomainType.UNKNOWN if call_dir == CallDirection.INBOUND else dom

    # Strict caller identity rule: If inbound, caller name cannot be a placeholder
    valid_name = caller_name if (call_dir == CallDirection.OUTBOUND or caller_name) else None
    if valid_name and valid_name.lower() in ["default caller", "unknown", "john doe"]:
        valid_name = None

    await manager.handle_media_stream(
        websocket=websocket,
        session_id=session_id,
        domain=initial_domain,
        direction=call_dir,
        caller_phone=caller_phone,
        caller_name=valid_name,
    )


@router.post("/api/v1/voice/session/start", response_model=SessionResponse)
async def start_session(
    payload: SessionStartRequest,
    engine: ConversationEngine = Depends(get_conversation_engine),
    state_manager: ConversationStateManager = Depends(get_state_manager),
) -> SessionResponse:
    """Explicitly initialize an inbound or outbound voice session."""
    direction = CallDirection.OUTBOUND if payload.direction.lower() == "outbound" else CallDirection.INBOUND
    raw_domain = payload.domain.lower()
    domain = DomainType.VAYVORA if raw_domain == "vayvora" else DomainType.EDUSAAS
    initial_domain = DomainType.UNKNOWN if direction == CallDirection.INBOUND else domain

    session_id = payload.session_id or f"session_{direction.value}_{int(time.time())}"

    # Strict identity check
    caller_name = payload.caller_name
    if caller_name and caller_name.lower() in ["default caller", "john doe"]:
        caller_name = None

    profile = CallerProfile(
        phone=payload.caller_phone,
        name=caller_name,
        email=None,
        campaign_objective=payload.campaign_objective,
    )

    state = state_manager.create_session(
        session_id=session_id,
        domain=initial_domain,
        direction=direction,
        caller_profile=profile,
    )

    opening_msg = None
    if direction == CallDirection.OUTBOUND:
        outbound_result = await engine.start_outbound_conversation(state)
        opening_msg = outbound_result.response_text

    return SessionResponse(
        session_id=session_id,
        domain=state.current_domain.value,
        direction=direction.value,
        conversation_active=state.conversation_active,
        current_stage=state.stage.value,
        caller_phone=state.caller.phone,
        caller_name=state.caller.name,
        turn_count=len(state.history),
        message=opening_msg,
        status="active" if state.conversation_active else "stopped",
    )


@router.post("/api/v1/voice/session/stop", response_model=SessionResponse)
@router.post("/api/v1/voice/session/{session_id}/stop", response_model=SessionResponse)
async def stop_session(
    session_id: Optional[str] = None,
    sid: Optional[str] = Query(default=None, alias="session_id"),
    reason: str = Query(default="caller_hung_up"),
    state_manager: ConversationStateManager = Depends(get_state_manager),
) -> SessionResponse:
    """Terminate an active voice session."""
    target_id = session_id or sid
    if not target_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    state = state_manager.get(target_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {target_id} not found.")

    state.conversation_active = False
    state.termination_reason = reason

    return SessionResponse(
        session_id=target_id,
        domain=state.current_domain.value,
        direction=state.metadata.direction.value,
        conversation_active=False,
        current_stage=state.stage.value,
        caller_phone=state.caller.phone,
        caller_name=state.caller.name,
        turn_count=len(state.history),
        message=f"Session terminated: {reason}",
        status="stopped",
    )


@router.get("/api/v1/voice/session/status", response_model=SessionResponse)
@router.get("/api/v1/voice/session/{session_id}/status", response_model=SessionResponse)
async def session_status(
    session_id: Optional[str] = None,
    sid: Optional[str] = Query(default=None, alias="session_id"),
    state_manager: ConversationStateManager = Depends(get_state_manager),
) -> SessionResponse:
    """Query current state and progress for a voice session."""
    target_id = session_id or sid
    if not target_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    state = state_manager.get(target_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {target_id} not found.")

    return SessionResponse(
        session_id=target_id,
        domain=state.current_domain.value,
        direction=state.metadata.direction.value,
        conversation_active=state.conversation_active,
        current_stage=state.stage.value,
        caller_phone=state.caller.phone,
        caller_name=state.caller.name,
        turn_count=len(state.history),
        status="active" if state.conversation_active else "stopped",
    )


@router.get("/api/v1/voice/sessions")
async def list_active_voice_sessions(
    manager: VoiceSessionManager = Depends(get_voice_session_manager),
    state_manager: ConversationStateManager = Depends(get_state_manager),
) -> Dict[str, Any]:
    """List IDs of active streaming voice sessions."""
    active = manager.list_active_sessions()
    sessions_dict = getattr(state_manager, "_sessions", {})
    all_sessions = list(sessions_dict.keys())
    combined = list(dict.fromkeys(active + all_sessions))
    return {
        "active_sessions": combined,
        "count": len(combined),
    }
