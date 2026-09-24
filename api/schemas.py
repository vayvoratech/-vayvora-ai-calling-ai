# C:\AI VOICE CALLING\api\schemas.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker: user or assistant")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    user_input: str = Field(..., description="Current query or transcription from user")
    session_id: Optional[str] = Field("default_session", description="Unique session identifier")
    tenant_id: Optional[str] = Field("default", description="Tenant or organization identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Prior conversation history")
    slots: Dict[str, Any] = Field(default_factory=dict, description="Extracted entity slots like date, time, name")


class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    route: str
    confidence: float
    margin: float
    rationale: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    slots: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ConfigResponse(BaseModel):
    model_name: str
    router_confidence_threshold: float
    router_temperature: float
    kb_resonance_weight: float
    tool_similarity_threshold: float