"""Pipecat Real-Time Voice Orchestration and Barge-In System.

Exports the VoicePipeline engine, VoiceSession lifecycle coordinator,
temporal state models, typed events, and provider adapters.
"""

from src.voice.adapters import (
    PipecatConversationAdapter,
    PipecatSTTAdapter,
    PipecatTTSAdapter,
    PipecatVADAdapter,
)
from src.voice.context import CancellationToken, TurnLifecycleState, VoiceDiagnostics
from src.voice.events import (
    AgentResponseCompleted,
    AgentResponseStarted,
    AgentThinking,
    CallerInterrupted,
    SessionCompleted,
    SpeechEnded,
    SpeechStarted,
    TranscriptFinal,
    TranscriptInterim,
    TTSStarted,
    TTSStopped,
    VoiceEvent,
    VoicePipelineErrorEvent,
)
from src.voice.pipeline import VoicePipeline
from src.voice.session import (
    MockAudioInput,
    MockAudioOutput,
    MockPipecatTransport,
    VoiceSession,
)

__all__ = [
    # Core Pipeline & Session
    "VoicePipeline",
    "VoiceSession",
    # Mocks
    "MockAudioInput",
    "MockAudioOutput",
    "MockPipecatTransport",
    # Context & States
    "TurnLifecycleState",
    "VoiceDiagnostics",
    "CancellationToken",
    # Adapters
    "PipecatVADAdapter",
    "PipecatSTTAdapter",
    "PipecatTTSAdapter",
    "PipecatConversationAdapter",
    # Events
    "VoiceEvent",
    "SpeechStarted",
    "SpeechEnded",
    "TranscriptInterim",
    "TranscriptFinal",
    "AgentThinking",
    "AgentResponseStarted",
    "AgentResponseCompleted",
    "TTSStarted",
    "TTSStopped",
    "CallerInterrupted",
    "SessionCompleted",
    "VoicePipelineErrorEvent",
]
