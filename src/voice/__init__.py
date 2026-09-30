"""Real-Time Voice Orchestration, Media Streaming, and Barge-In System.

Exports the VoicePipeline engine, VoiceSession lifecycle coordinator,
VoiceSessionManager, AudioProcessor, VADService, STTService, DeepgramFluxTTS,
temporal state models, typed events, and provider adapters.
"""

from src.voice.adapters import (
    PipecatConversationAdapter,
    PipecatSTTAdapter,
    PipecatTTSAdapter,
    PipecatVADAdapter,
)
from src.voice.audio_services.audio_processor import AudioProcessor
from src.voice.audio_services.vad_services import VADEvent, VADEventType, VADService
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
from src.voice.session_manager import VoiceSessionManager
from src.voice.stt.stt_service import GroqWhisperSTT, STTService
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS, DeepgramTTS

__all__ = [
    # Core Pipeline, Manager & Session
    "VoicePipeline",
    "VoiceSession",
    "VoiceSessionManager",
    "AudioProcessor",
    "VADService",
    "VADEvent",
    "VADEventType",
    "STTService",
    "GroqWhisperSTT",
    "DeepgramFluxTTS",
    "DeepgramTTS",
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
