"""Context models, turn lifecycle states, diagnostics, and cancellation tokens.

Tracks pipeline timing, measured latencies, generation IDs, and race condition
guards across real-time voice turns.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional


class TurnLifecycleState(str, Enum):
    """Real-time voice turn lifecycle states for Pipecat pipeline coordination."""

    LISTENING = "LISTENING"
    SPEECH_DETECTED = "SPEECH_DETECTED"
    CAPTURING = "CAPTURING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class CancellationToken:
    """Thread-safe cancellation token and generation tag for invalidating stale outputs."""

    def __init__(self, generation_id: int = 0) -> None:
        self.generation_id: int = generation_id
        self._is_cancelled: bool = False

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def cancel(self) -> None:
        """Flag the current generation as cancelled (e.g. on barge-in)."""
        self._is_cancelled = True

    def reset(self, new_generation_id: int) -> None:
        """Advance generation ID and reset cancellation token for a fresh turn."""
        self.generation_id = new_generation_id
        self._is_cancelled = False

    def is_valid_generation(self, gen_id: int) -> bool:
        """Verify whether an incoming frame/result belongs to the active, uncancelled generation."""
        return not self._is_cancelled and gen_id == self.generation_id


@dataclass
class VoiceDiagnostics:
    """Measured operational latency, RTF, and counters for developer inspection."""

    session_id: str
    turn_id: int = 0
    current_generation_id: int = 0

    # VAD Measurements
    speech_start_time: Optional[float] = None
    speech_end_time: Optional[float] = None
    speech_duration: float = 0.0

    # STT Measurements
    transcription_latency: Optional[float] = None
    audio_duration: float = 0.0
    stt_rtf: Optional[float] = None

    # LLM Measurements
    first_token_latency: Optional[float] = None
    total_generation_latency: Optional[float] = None

    # TTS Measurements
    first_audio_latency: Optional[float] = None
    total_synthesis_latency: Optional[float] = None
    tts_rtf: Optional[float] = None

    # Pipeline Turn & Barge-In Metrics
    turn_latency: Optional[float] = None
    interruption_count: int = 0
    cancelled_generations_count: int = 0
    dropped_stale_frames_count: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize diagnostics for UI and telemetry reporting."""
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "current_generation_id": self.current_generation_id,
            "vad": {
                "speech_start_time": self.speech_start_time,
                "speech_end_time": self.speech_end_time,
                "speech_duration": round(self.speech_duration, 4),
            },
            "stt": {
                "transcription_latency": round(self.transcription_latency, 4) if self.transcription_latency else None,
                "audio_duration": round(self.audio_duration, 4),
                "rtf": self.stt_rtf,
            },
            "llm": {
                "first_token_latency": round(self.first_token_latency, 4) if self.first_token_latency else None,
                "total_generation_latency": round(self.total_generation_latency, 4) if self.total_generation_latency else None,
            },
            "tts": {
                "first_audio_latency": round(self.first_audio_latency, 4) if self.first_audio_latency else None,
                "total_synthesis_latency": round(self.total_synthesis_latency, 4) if self.total_synthesis_latency else None,
                "rtf": self.tts_rtf,
            },
            "pipeline": {
                "turn_latency": round(self.turn_latency, 4) if self.turn_latency else None,
                "interruption_count": self.interruption_count,
                "cancelled_generations_count": self.cancelled_generations_count,
                "dropped_stale_frames_count": self.dropped_stale_frames_count,
                "errors": list(self.errors),
            },
        }
