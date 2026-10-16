"""Turn latency instrumentation and telemetry tracking.

Measures precise monotonic timestamps for voice agent turns:
T0 = speech starts
T1 = VAD speech end
T2 = STT request start
T3 = STT transcript received
T4 = Gemini decision start
T5 = Gemini decision received
T6 = RAG start
T7 = RAG complete
T8 = MCP/tool start
T9 = MCP/tool complete
T10 = final response ready
T11 = Deepgram TTS request/start
T12 = first TTS audio received
T13 = first audio sent to client

Most critical metric: speech_end -> first_audio (T1 -> T13).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.logging import get_logger

logger = get_logger("voice.latency")


@dataclass
class TurnLatencyTracker:
    """Monotonic timestamp tracker for a discrete voice turn."""

    turn_id: int = 1
    session_id: str = ""

    # Monotonic timestamps
    t0_speech_start: Optional[float] = None
    t1_speech_end: Optional[float] = None
    t2_stt_start: Optional[float] = None
    t3_stt_end: Optional[float] = None
    t4_decision_start: Optional[float] = None
    t5_decision_end: Optional[float] = None
    t6_rag_start: Optional[float] = None
    t7_rag_end: Optional[float] = None
    t8_tool_start: Optional[float] = None
    t9_tool_end: Optional[float] = None
    t10_response_ready: Optional[float] = None
    t11_tts_start: Optional[float] = None
    t12_tts_first_audio: Optional[float] = None
    t13_first_audio_sent: Optional[float] = None

    # Streaming monotonic timestamps (Phase 5C)
    t_grounded_synth_start: Optional[float] = None
    t_first_grounded_phrase: Optional[float] = None
    t_grounded_gen_end: Optional[float] = None
    t_tts_complete: Optional[float] = None

    # Contextual metadata
    domain: str = ""
    intent: str = ""
    rag_invoked: bool = False
    tool_invoked: bool = False
    reused_first_response: bool = False

    def mark_speech_start(self, ts: Optional[float] = None) -> None:
        self.t0_speech_start = ts or time.perf_counter()

    def mark_speech_end(self, ts: Optional[float] = None) -> None:
        self.t1_speech_end = ts or time.perf_counter()

    def mark_stt_start(self, ts: Optional[float] = None) -> None:
        self.t2_stt_start = ts or time.perf_counter()

    def mark_stt_end(self, ts: Optional[float] = None) -> None:
        self.t3_stt_end = ts or time.perf_counter()

    def mark_decision_start(self, ts: Optional[float] = None) -> None:
        self.t4_decision_start = ts or time.perf_counter()

    def mark_decision_end(self, ts: Optional[float] = None) -> None:
        self.t5_decision_end = ts or time.perf_counter()

    def mark_rag_start(self, ts: Optional[float] = None) -> None:
        self.t6_rag_start = ts or time.perf_counter()
        self.rag_invoked = True

    def mark_rag_end(self, ts: Optional[float] = None) -> None:
        self.t7_rag_end = ts or time.perf_counter()

    def mark_tool_start(self, ts: Optional[float] = None) -> None:
        self.t8_tool_start = ts or time.perf_counter()
        self.tool_invoked = True

    def mark_tool_end(self, ts: Optional[float] = None) -> None:
        self.t9_tool_end = ts or time.perf_counter()

    def mark_response_ready(self, ts: Optional[float] = None) -> None:
        self.t10_response_ready = ts or time.perf_counter()

    def mark_tts_start(self, ts: Optional[float] = None) -> None:
        self.t11_tts_start = ts or time.perf_counter()

    def mark_tts_first_audio(self, ts: Optional[float] = None) -> None:
        if self.t12_tts_first_audio is None:
            self.t12_tts_first_audio = ts or time.perf_counter()

    def mark_first_audio_sent(self, ts: Optional[float] = None) -> None:
        if self.t13_first_audio_sent is None:
            self.t13_first_audio_sent = ts or time.perf_counter()

    def mark_grounded_synth_start(self, ts: Optional[float] = None) -> None:
        self.t_grounded_synth_start = ts or time.perf_counter()

    def mark_first_grounded_phrase(self, ts: Optional[float] = None) -> None:
        if self.t_first_grounded_phrase is None:
            self.t_first_grounded_phrase = ts or time.perf_counter()

    def mark_grounded_gen_end(self, ts: Optional[float] = None) -> None:
        self.t_grounded_gen_end = ts or time.perf_counter()

    def mark_tts_complete(self, ts: Optional[float] = None) -> None:
        self.t_tts_complete = ts or time.perf_counter()

    @property
    def vad_latency_ms(self) -> float:
        if self.t0_speech_start and self.t1_speech_end:
            return round((self.t1_speech_end - self.t0_speech_start) * 1000, 2)
        return 0.0

    @property
    def stt_latency_ms(self) -> float:
        if self.t2_stt_start and self.t3_stt_end:
            return round((self.t3_stt_end - self.t2_stt_start) * 1000, 2)
        return 0.0

    @property
    def gemini_latency_ms(self) -> float:
        if self.t4_decision_start and self.t5_decision_end:
            return round((self.t5_decision_end - self.t4_decision_start) * 1000, 2)
        return 0.0

    @property
    def rag_latency_ms(self) -> float:
        if self.t6_rag_start and self.t7_rag_end:
            return round((self.t7_rag_end - self.t6_rag_start) * 1000, 2)
        return 0.0

    @property
    def tool_latency_ms(self) -> float:
        if self.t8_tool_start and self.t9_tool_end:
            return round((self.t9_tool_end - self.t8_tool_start) * 1000, 2)
        return 0.0

    @property
    def tts_first_audio_latency_ms(self) -> float:
        if self.t11_tts_start and self.t12_tts_first_audio:
            return round((self.t12_tts_first_audio - self.t11_tts_start) * 1000, 2)
        return 0.0

    @property
    def total_time_to_first_audio_ms(self) -> float:
        """Measure speech_end / stt_end -> first_audio sent (or first audio produced)."""
        ref_end = self.t13_first_audio_sent or self.t12_tts_first_audio or self.t10_response_ready
        ref_start = self.t3_stt_end or self.t1_speech_end or self.t2_stt_start or self.t4_decision_start
        if ref_start and ref_end and ref_end >= ref_start:
            return round((ref_end - ref_start) * 1000, 2)
        return 0.0

    @property
    def time_to_first_grounded_phrase_ms(self) -> float:
        """Measure grounded synthesis start -> first grounded phrase emitted."""
        if self.t_first_grounded_phrase and self.t_grounded_synth_start:
            return round((self.t_first_grounded_phrase - self.t_grounded_synth_start) * 1000, 2)
        return 0.0

    @property
    def tts_phrase_to_first_audio_ms(self) -> float:
        """Measure first grounded phrase -> first audio sent (TTS delivery latency)."""
        ref_phrase = self.t_first_grounded_phrase or self.t11_tts_start
        ref_audio = self.t13_first_audio_sent or self.t12_tts_first_audio
        if ref_phrase and ref_audio and ref_audio >= ref_phrase:
            return round((ref_audio - ref_phrase) * 1000, 2)
        return 0.0

    @property
    def streaming_overlap(self) -> bool:
        """True if first audio frame was delivered to client before grounded text generation concluded."""
        return bool(
            self.t13_first_audio_sent
            and self.t_grounded_gen_end
            and self.t13_first_audio_sent < self.t_grounded_gen_end
        )

    @property
    def total_response_time_ms(self) -> float:
        """Total turn time from user speech/turn start to TTS audio completion."""
        ref_start = self.t3_stt_end or self.t1_speech_end or self.t2_stt_start or self.t4_decision_start
        ref_end = self.t_tts_complete or self.t13_first_audio_sent or self.t10_response_ready
        if ref_start and ref_end and ref_end >= ref_start:
            return round((ref_end - ref_start) * 1000, 2)
        return 0.0

    # Monotonic benchmark timestamp aliases (T0 - T9)
    @property
    def t0_turn_start(self) -> Optional[float]:
        return self.t3_stt_end or self.t1_speech_end or self.t4_decision_start

    @property
    def t1_decision_start(self) -> Optional[float]:
        return self.t4_decision_start

    @property
    def t2_decision_end(self) -> Optional[float]:
        return self.t5_decision_end

    @property
    def t3_retrieval_start(self) -> Optional[float]:
        return self.t6_rag_start

    @property
    def t4_retrieval_end(self) -> Optional[float]:
        return self.t7_rag_end

    @property
    def t5_grounded_synth_start(self) -> Optional[float]:
        return self.t_grounded_synth_start

    @property
    def t6_first_grounded_phrase(self) -> Optional[float]:
        return self.t_first_grounded_phrase

    @property
    def t7_first_audio_sent(self) -> Optional[float]:
        return self.t13_first_audio_sent

    @property
    def t8_grounded_gen_end(self) -> Optional[float]:
        return self.t_grounded_gen_end

    @property
    def t9_playback_complete(self) -> Optional[float]:
        return self.t_tts_complete

    def to_dict(self) -> Dict[str, Any]:
        """Export serialized latency metrics dict."""
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "domain": self.domain,
            "intent": self.intent,
            "rag_invoked": self.rag_invoked,
            "tool_invoked": self.tool_invoked,
            "reused_first_response": self.reused_first_response,
            "vad_latency_ms": self.vad_latency_ms,
            "stt_latency_ms": self.stt_latency_ms,
            "gemini_latency_ms": self.gemini_latency_ms,
            "rag_latency_ms": self.rag_latency_ms,
            "tool_latency_ms": self.tool_latency_ms,
            "tts_first_audio_latency_ms": self.tts_first_audio_latency_ms,
            "tts_phrase_to_first_audio_ms": self.tts_phrase_to_first_audio_ms,
            "speech_end_to_first_audio_ms": self.total_time_to_first_audio_ms,
            "time_to_first_grounded_phrase_ms": self.time_to_first_grounded_phrase_ms,
            "streaming_overlap": self.streaming_overlap,
            "total_response_time_ms": self.total_response_time_ms,
            "t0_turn_start": self.t0_turn_start,
            "t1_decision_start": self.t1_decision_start,
            "t2_decision_end": self.t2_decision_end,
            "t3_retrieval_start": self.t3_retrieval_start,
            "t4_retrieval_end": self.t4_retrieval_end,
            "t5_grounded_synth_start": self.t5_grounded_synth_start,
            "t6_first_grounded_phrase": self.t6_first_grounded_phrase,
            "t7_first_audio_sent": self.t7_first_audio_sent,
            "t8_grounded_gen_end": self.t8_grounded_gen_end,
            "t9_playback_complete": self.t9_playback_complete,
        }

    def log_summary(self) -> None:
        """Emit clean single-line latency summary at INFO level."""
        logger.info(
            "LATENCY SUMMARY [Turn %d | %s] | STT: %.1fms | LLM: %.1fms | RAG: %.1fms | Tool: %.1fms | TTS 1st: %.1fms | SpeechEnd->1stAudio: %.1fms",
            self.turn_id,
            self.domain or "unknown",
            self.stt_latency_ms,
            self.gemini_latency_ms,
            self.rag_latency_ms,
            self.tool_latency_ms,
            self.tts_first_audio_latency_ms,
            self.total_time_to_first_audio_ms,
        )
