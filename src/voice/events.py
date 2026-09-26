"""Typed event models for real-time voice orchestration tracking.

Provides immutable, structured event objects capturing speech boundaries,
transcripts, LLM streaming chunks, TTS synthesis markers, barge-in interruptions,
tool dispatches, and pipeline diagnostics.
Audio byte buffers are intentionally omitted to keep event streams lightweight.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class VoiceEvent:
    """Base event model for voice pipeline lifecycle."""

    session_id: str
    timestamp: float = field(default_factory=time.time)
    turn_id: Optional[int] = None
    generation_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpeechStarted(VoiceEvent):
    """Fired when VAD detects the onset of caller speech."""

    pass


@dataclass(frozen=True)
class SpeechEnded(VoiceEvent):
    """Fired when VAD detects the termination of caller speech after min_silence_duration."""

    speech_duration: float = 0.0


@dataclass(frozen=True)
class TranscriptInterim(VoiceEvent):
    """Fired when STT emits a provisional, non-final transcription chunk."""

    transcript: str = ""
    confidence: Optional[float] = None


@dataclass(frozen=True)
class TranscriptFinal(VoiceEvent):
    """Fired when STT emits a finalized user utterance transcription."""

    transcript: str = ""
    confidence: Optional[float] = None
    audio_duration: float = 0.0
    latency: float = 0.0
    real_time_factor: Optional[float] = None


@dataclass(frozen=True)
class AgentThinking(VoiceEvent):
    """Fired when ConversationEngine begins processing a user turn."""

    pass


@dataclass(frozen=True)
class AgentResponseStarted(VoiceEvent):
    """Fired when the conversational agent decision or first text token emerges."""

    pass


@dataclass(frozen=True)
class AgentResponseChunk(VoiceEvent):
    """Fired when an incremental text token or sentence chunk is emitted for TTS."""

    chunk_text: str = ""


@dataclass(frozen=True)
class AgentResponseCompleted(VoiceEvent):
    """Fired when the entire agent conversational response has been synthesized."""

    full_text: str = ""
    domain: str = ""
    intent: str = ""
    stage: str = ""


@dataclass(frozen=True)
class TTSStarted(VoiceEvent):
    """Fired when TTS audio synthesis and playback commences."""

    voice: str = ""
    sample_rate: int = 24000


@dataclass(frozen=True)
class TTSStopped(VoiceEvent):
    """Fired when TTS playback completes naturally."""

    duration: float = 0.0
    real_time_factor: Optional[float] = None


@dataclass(frozen=True)
class CallerInterrupted(VoiceEvent):
    """Fired when caller speaks over the active agent utterance (barge-in)."""

    cancelled_generation_id: Optional[int] = None
    reason: str = "caller_speech_detected"


@dataclass(frozen=True)
class ToolStarted(VoiceEvent):
    """Fired when an external MCP action is dispatched."""

    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCompleted(VoiceEvent):
    """Fired when an external action returns successfully and is verified."""

    tool_name: str = ""
    verified: bool = True
    reference: Optional[str] = None


@dataclass(frozen=True)
class ToolFailed(VoiceEvent):
    """Fired when an external action fails verification or provider error."""

    tool_name: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class SessionCompleted(VoiceEvent):
    """Fired when the voice call concludes and session closes."""

    reason: str = "completed"
    total_turns: int = 0
    total_interruptions: int = 0


@dataclass(frozen=True)
class VoicePipelineErrorEvent(VoiceEvent):
    """Fired when an unexpected error occurs in frame routing or transport."""

    error_message: str = ""
    component: str = "pipeline"
