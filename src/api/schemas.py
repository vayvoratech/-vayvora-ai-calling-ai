"""Pydantic request and response schemas for REST and WebSocket API endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of speaker: 'caller'/'user' or 'agent'/'assistant'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    user_input: Optional[str] = Field(default=None, description="Current query or transcription from caller")
    message: Optional[str] = Field(default=None, description="Alternative alias for user_input")
    session_id: str = Field(default="default_session", description="Unique session identifier")
    domain: Optional[str] = Field(default="edusaas", description="Domain: 'edusaas' or 'vayvora'")
    tenant_id: Optional[str] = Field(default="default", description="Tenant or organization identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Prior conversation history")
    slots: Dict[str, Any] = Field(default_factory=dict, description="Extracted entity slots")

    @property
    def query_text(self) -> str:
        return (self.user_input or self.message or "").strip()


class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    route: str
    confidence: float
    margin: float = 0.0
    rationale: str = ""
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    slots: Dict[str, Any] = Field(default_factory=dict)
    grounded_citations: List[str] = Field(default_factory=list)
    stage: str = "information"
    conversation_active: bool = True
    error: Optional[str] = None
    session_id: Optional[str] = None
    domain: Optional[str] = None


class ConfigResponse(BaseModel):
    model: str = "gemini-3.5-flash-lite"
    model_name: str = "gemini-3.5-flash-lite"
    llm_provider: str
    default_domain: str = "edusaas"
    supported_domains: List[str] = Field(default_factory=lambda: ["edusaas", "vayvora"])
    audio_sample_rate: int = 16000
    relevance_threshold: float
    stt_provider: str
    tts_provider: str
    vad_provider: str


class SessionStartRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Optional custom session ID")
    domain: str = Field(default="edusaas", description="Domain: 'edusaas' or 'vayvora'")
    direction: str = Field(default="inbound", description="Call direction: 'inbound' or 'outbound'")
    caller_phone: Optional[str] = Field(default=None, description="Caller phone number (if verified)")
    caller_name: Optional[str] = Field(default=None, description="Caller name (if verified)")
    campaign_objective: Optional[str] = Field(default=None, description="Outbound campaign objective")


class SessionResponse(BaseModel):
    session_id: str
    domain: str
    direction: str
    conversation_active: bool
    current_stage: str
    caller_phone: Optional[str] = None
    caller_name: Optional[str] = None
    turn_count: int = 0
    message: Optional[str] = None
    status: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float
    subsystems: Dict[str, Any] = Field(default_factory=dict)
    components: Dict[str, Any] = Field(default_factory=dict)
