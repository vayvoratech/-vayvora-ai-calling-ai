"""Phase 5C -- Phase 4: End-to-End Voice Latency Benchmark Runner.

Instruments and measures monotonic timestamps across:
- T0: voice-turn processing begins (STT final / speech end)
- T1: decision prompt start
- T2: decision prompt complete / decision parsed
- T3: retrieval start
- T4: retrieval complete
- T5: grounded synthesis start
- T6: first grounded token/phrase emitted
- T7: first audio chunk transmitted to client/mock client
- T8: grounded generation complete
- T9: final audio chunk transmitted / playback queue complete

Calculates:
- TTFA (Time-to-First-Audio): T7 - T0
- Time to first grounded phrase: T6 - T0 (and T6 - T5)
- TTS latency: T7 - T6
- Total turn response time: T9 - T0
- Streaming overlap: T7 < T8 (True/False)

Fixed Query Set:
1. Grounded project/amenities: "Tell me about the plots and amenities at Pristine City"
2. Pricing with verified evidence: "What is the price of plots at Pristine City?"
3. Insufficient RAG evidence: "Tell me about the swimming pool on Mars"
4. Conversational non-RAG turn: "Good morning, how are you today?"
5. Tool-driven turn: "Please schedule a site visit for Pristine City tomorrow at 10 AM"

Execution Modes:
- Mock Benchmark: 10 iterations per query per mode (Streaming vs Baseline)
- Live Benchmark: Bounded live runs with live Gemini & Deepgram TTS
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings, get_settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine, EngineTurnResult
from src.core.interfaces import KnowledgeProvider, LLMProvider, ToolProvider
from src.core.llm import GeminiLLMProvider, MockLLMProvider
from src.core.types import (
    CallDirection,
    ConversationStage,
    DomainType,
    RAGChunk,
    RAGQuery,
    TurnRole,
)
from src.rag.embeddings import MockEmbeddingProvider, get_embedding_provider
from src.rag.ingestion import KnowledgeIngestionPipeline
from src.rag.retriever import GroundedContextResult, GroundedKnowledgeProvider
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.tools.mcp_client import MockToolProvider
from src.voice.latency_tracker import TurnLatencyTracker
from src.voice.session_manager import VoiceSessionManager
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS


# =============================================================================
# Benchmark Mock Infrastructure
# =============================================================================

class BenchmarkMockWebSocket:
    """Mock WebSocket capturing sent JSON and binary frames."""

    def __init__(self):
        self.sent_json: List[Dict[str, Any]] = []
        self.sent_bytes: List[bytes] = []
        self.first_audio_sent_ts: Optional[float] = None
        self.final_audio_sent_ts: Optional[float] = None

    async def send_text(self, text: str) -> None:
        try:
            self.sent_json.append(json.loads(text))
        except Exception:
            self.sent_json.append({"_raw": text})

    async def send_bytes(self, data: bytes) -> None:
        now = time.perf_counter()
        if self.first_audio_sent_ts is None:
            self.first_audio_sent_ts = now
        self.final_audio_sent_ts = now
        self.sent_bytes.append(data)

    async def close(self) -> None:
        pass


class BenchmarkMockTTS(DeepgramFluxTTS):
    """Mock Deepgram TTS simulating realistic per-chunk audio delivery."""

    def __init__(self, frame_delay_s: float = 0.005):
        super().__init__(api_key="")
        self.frame_delay_s = frame_delay_s
        self.sent_phrases: List[str] = []
        self._mock_mode = True
        self.connected = True

    async def connect(self, *args, **kwargs) -> None:
        self.connected = True
        self._mock_mode = True

    async def send_text(self, text: str) -> None:
        now = time.perf_counter()
        self.sent_phrases.append(text)
        if self.frame_delay_s > 0:
            await asyncio.sleep(self.frame_delay_s)

        mock_pcm = b"\x00\x00" * 160  # 10ms of 16kHz silence
        if self.audio_callback:
            res = self.audio_callback(mock_pcm)
            if asyncio.iscoroutine(res):
                await res

        if not self.first_audio_received:
            self.first_audio_received = True
            if self.event_callback:
                res = self.event_callback({
                    "type": "FirstAudio",
                    "timestamp": now,
                    "audio_bytes": len(mock_pcm),
                })
                if asyncio.iscoroutine(res):
                    await res

    async def flush(self) -> None:
        if self.event_callback:
            res = self.event_callback({"type": "Flushed", "timestamp": time.perf_counter()})
            if asyncio.iscoroutine(res):
                await res

    def reset_turn(self) -> None:
        super().reset_turn()
        self.sent_phrases.clear()


class BenchmarkMockLLM(MockLLMProvider):
    """Mock LLM supporting both batch decisions and streaming fragment emission with token pacing."""

    def __init__(
        self,
        canned_decisions: Optional[List[ConversationalDecision]] = None,
        canned_streams: Optional[List[List[str]]] = None,
        fragment_delay_s: float = 0.02,
    ):
        super().__init__(canned_decisions=canned_decisions)
        self.canned_streams = canned_streams or []
        self.fragment_delay_s = fragment_delay_s

    async def stream_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[str]:
        if self.canned_streams:
            fragments = self.canned_streams.pop(0)
            for frag in fragments:
                if self.fragment_delay_s > 0:
                    await asyncio.sleep(self.fragment_delay_s)
                yield frag
        else:
            default_chunks = ["Pristine City offers ", "gated villa plots ", "with 24/7 security, ", "wide roads, ", "and landscaped gardens."]
            for frag in default_chunks:
                if self.fragment_delay_s > 0:
                    await asyncio.sleep(self.fragment_delay_s)
                yield frag


# =============================================================================
# Benchmark Data Structures
# =============================================================================

@dataclass
class BenchmarkTurnRecord:
    env: str  # "MOCK" or "LIVE"
    mode: str  # "STREAMING" or "BASELINE"
    query_id: int
    query_label: str
    iteration: int
    user_utterance: str
    response_text: str
    # Monotonic timestamps
    t0_turn_start: float
    t1_decision_start: Optional[float]
    t2_decision_end: Optional[float]
    t3_retrieval_start: Optional[float]
    t4_retrieval_end: Optional[float]
    t5_grounded_synth_start: Optional[float]
    t6_first_grounded_phrase: Optional[float]
    t7_first_audio_sent: Optional[float]
    t8_grounded_gen_end: Optional[float]
    t9_playback_complete: Optional[float]
    # Calculated latencies in milliseconds
    ttfa_ms: float
    time_to_first_grounded_phrase_ms: float
    tts_latency_ms: float
    total_response_time_ms: float
    streaming_overlap: bool


BENCHMARK_QUERIES = [
    {
        "id": 1,
        "label": "Grounded Project/Amenities",
        "text": "Tell me about the plots and amenities at Pristine City",
    },
    {
        "id": 2,
        "label": "Pricing with Verified Evidence",
        "text": "What is the price of plots at Pristine City?",
    },
    {
        "id": 3,
        "label": "Insufficient RAG Evidence",
        "text": "Tell me about the swimming pool on Mars",
    },
    {
        "id": 4,
        "label": "Conversational Non-RAG",
        "text": "Good morning, how are you today?",
    },
    {
        "id": 5,
        "label": "Tool-Driven Turn",
        "text": "Please schedule a site visit for Pristine City tomorrow at 10 AM",
    },
]


# =============================================================================
# Mock Benchmark Runner
# =============================================================================

async def run_single_mock_turn(
    query_info: Dict[str, Any],
    iteration: int,
    streaming: bool,
) -> BenchmarkTurnRecord:
    qid = query_info["id"]
    qlabel = query_info["label"]
    utterance = query_info["text"]

    chunk = RAGChunk(
        doc_id="infra:pristine_city:overview",
        domain=DomainType.INFRA_BUILDER,
        title="Pristine City Overview",
        content="Pristine City is an exclusive plotted layout with 40-foot wide blacktop roads, clubhouse, and underground drainage. Pricing is INR 3,500 per sq yd.",
        score=0.95,
        metadata={"project": "Pristine City"},
    )

    # Configure canned decision & stream based on query type
    if qid == 1:
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="",
            knowledge_required=True,
            knowledge_query="Pristine City plots amenities",
        )
        stream_frags = ["Pristine City offers luxury gated plots ", "with wide blacktop roads, ", "underground drainage, ", "and an exclusive clubhouse."]
        batch_text = "Pristine City offers luxury gated plots with wide blacktop roads, underground drainage, and an exclusive clubhouse."
        chunks = [chunk]
    elif qid == 2:
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="price_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="",
            knowledge_required=True,
            knowledge_query="Pristine City price per sq yd",
        )
        stream_frags = ["The verified price for Pristine City ", "is INR 3,500 per sq yd."]
        batch_text = "The verified price for Pristine City is INR 3,500 per sq yd."
        chunks = [chunk]
    elif qid == 3:
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="property_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="",
            knowledge_required=True,
            knowledge_query="Mars swimming pool",
        )
        stream_frags = []
        batch_text = "I checked our knowledge base, but I don't have verified details regarding that at the moment."
        chunks = []  # No chunks found
    elif qid == 4:
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="greeting",
            proposed_stage=ConversationStage.GREETING,
            user_facing_response="Good morning! I'm doing well, thank you. How can I assist you with your property search today?",
            knowledge_required=False,
        )
        stream_frags = []
        batch_text = decision.user_facing_response
        chunks = []
    else:  # Tool-driven
        decision = ConversationalDecision(
            detected_domain=DomainType.INFRA_BUILDER,
            detected_intent="schedule_site_visit",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            user_facing_response="I would be happy to schedule a site visit for Pristine City tomorrow at 10 AM. May I confirm your name and phone number?",
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="schedule_site_visit",
                parameters={"project": "Pristine City", "datetime": "tomorrow at 10 AM"},
            ),
            knowledge_required=False,
        )
        stream_frags = []
        batch_text = decision.user_facing_response
        chunks = []

    mock_llm = BenchmarkMockLLM(
        canned_decisions=[decision],
        canned_streams=[list(stream_frags)] if stream_frags else None,
        fragment_delay_s=0.03,
    )
    if not streaming:
        mock_llm.stream_response = None  # Force batch generation

    mock_kp = GroundedKnowledgeProvider(
        embedding_provider=MockEmbeddingProvider(dimension=64),
        in_memory=True,
    )
    for c in chunks:
        await mock_kp.index_document(c)

    engine = ConversationEngine(llm_provider=mock_llm, knowledge_provider=mock_kp, tool_provider=MockToolProvider())
    state_mgr = ConversationStateManager()
    session_id = f"bench-mock-{qid}-{iteration}-{'stream' if streaming else 'batch'}"
    state = state_mgr.create_session(session_id, DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    tts = BenchmarkMockTTS(frame_delay_s=0.005)
    ws = BenchmarkMockWebSocket()
    tts.audio_callback = ws.send_bytes
    tts_done_event = asyncio.Event()

    async def event_cb(ev: dict):
        if ev.get("type") in ("Flushed", "SpeechMetadata"):
            tts_done_event.set()

    tts.event_callback = event_cb

    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    tracker = TurnLatencyTracker(turn_id=1, session_id=session_id, domain="infra_builder")
    t0 = time.perf_counter()
    tracker.mark_speech_end(t0)
    tracker.mark_stt_end(t0)

    await voice_mgr._process_agent_turn(
        websocket=ws,
        ws_lock=asyncio.Lock(),
        state=state,
        transcript=utterance,
        deepgram_tts=tts,
        connection_alive=lambda: True,
        tracker=tracker,
        turn_id=1,
        tts_done_event=tts_done_event,
        enable_grounded_streaming=streaming,
    )

    turn_metrics = tracker.to_dict()
    # Pull audio sent timestamps from WS if not recorded
    t7 = tracker.t7_first_audio_sent or ws.first_audio_sent_ts
    t9 = tracker.t9_playback_complete or ws.final_audio_sent_ts

    ttfa = round((t7 - t0) * 1000, 2) if (t7 and t7 >= t0) else 0.0
    phrase_lat = round((tracker.t6_first_grounded_phrase - t0) * 1000, 2) if (tracker.t6_first_grounded_phrase and tracker.t6_first_grounded_phrase >= t0) else 0.0
    tts_lat = round((t7 - tracker.t6_first_grounded_phrase) * 1000, 2) if (t7 and tracker.t6_first_grounded_phrase and t7 >= tracker.t6_first_grounded_phrase) else 0.0
    total_time = round((t9 - t0) * 1000, 2) if (t9 and t9 >= t0) else 0.0
    overlap = bool(t7 and tracker.t8_grounded_gen_end and t7 < tracker.t8_grounded_gen_end)

    return BenchmarkTurnRecord(
        env="MOCK",
        mode="STREAMING" if streaming else "BASELINE",
        query_id=qid,
        query_label=qlabel,
        iteration=iteration,
        user_utterance=utterance,
        response_text=state.history[-1].content if state.history else "",
        t0_turn_start=t0,
        t1_decision_start=tracker.t1_decision_start,
        t2_decision_end=tracker.t2_decision_end,
        t3_retrieval_start=tracker.t3_retrieval_start,
        t4_retrieval_end=tracker.t4_retrieval_end,
        t5_grounded_synth_start=tracker.t5_grounded_synth_start,
        t6_first_grounded_phrase=tracker.t6_first_grounded_phrase,
        t7_first_audio_sent=t7,
        t8_grounded_gen_end=tracker.t8_grounded_gen_end,
        t9_playback_complete=t9,
        ttfa_ms=ttfa,
        time_to_first_grounded_phrase_ms=phrase_lat,
        tts_latency_ms=tts_lat,
        total_response_time_ms=total_time,
        streaming_overlap=overlap,
    )


# =============================================================================
# Live Benchmark Runner
# =============================================================================

async def run_single_live_turn(
    query_info: Dict[str, Any],
    iteration: int,
    streaming: bool,
    engine: ConversationEngine,
    state_mgr: ConversationStateManager,
    settings: Settings,
) -> BenchmarkTurnRecord:
    qid = query_info["id"]
    qlabel = query_info["label"]
    utterance = query_info["text"]

    session_id = f"bench-live-{qid}-{iteration}-{'stream' if streaming else 'batch'}"
    state = state_mgr.create_session(session_id, DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    deepgram_key = settings.deepgram_api_key.get_secret_value() if settings.deepgram_api_key else ""
    tts = DeepgramFluxTTS(api_key=deepgram_key)
    ws = BenchmarkMockWebSocket()
    tts_done_event = asyncio.Event()

    async def audio_cb(pcm_bytes: bytes):
        await ws.send_bytes(pcm_bytes)

    async def event_cb(ev: dict):
        if ev.get("type") in ("Flushed", "SpeechMetadata"):
            tts_done_event.set()

    tts.audio_callback = audio_cb
    tts.event_callback = event_cb

    try:
        await tts.connect()
    except Exception as exc:
        print(f"Warning: Live Deepgram connect issue ({exc}); falling back to mock TTS mode")
        tts._mock_mode = True

    voice_mgr = VoiceSessionManager(engine=engine, state_manager=state_mgr, tts_service=lambda: tts)
    voice_mgr._active_sessions[session_id] = {
        "websocket": ws,
        "state": state,
        "tts": tts,
        "active_turn_id": 1,
    }

    tracker = TurnLatencyTracker(turn_id=1, session_id=session_id, domain="infra_builder")
    t0 = time.perf_counter()
    tracker.mark_speech_end(t0)
    tracker.mark_stt_end(t0)

    try:
        await voice_mgr._process_agent_turn(
            websocket=ws,
            ws_lock=asyncio.Lock(),
            state=state,
            transcript=utterance,
            deepgram_tts=tts,
            connection_alive=lambda: True,
            tracker=tracker,
            turn_id=1,
            tts_done_event=tts_done_event,
            enable_grounded_streaming=streaming,
        )
    finally:
        try:
            await tts.close()
        except Exception:
            pass

    t7 = tracker.t7_first_audio_sent or ws.first_audio_sent_ts
    t9 = tracker.t9_playback_complete or ws.final_audio_sent_ts

    ttfa = round((t7 - t0) * 1000, 2) if (t7 and t7 >= t0) else 0.0
    phrase_lat = round((tracker.t6_first_grounded_phrase - t0) * 1000, 2) if (tracker.t6_first_grounded_phrase and tracker.t6_first_grounded_phrase >= t0) else 0.0
    tts_lat = round((t7 - tracker.t6_first_grounded_phrase) * 1000, 2) if (t7 and tracker.t6_first_grounded_phrase and t7 >= tracker.t6_first_grounded_phrase) else 0.0
    total_time = round((t9 - t0) * 1000, 2) if (t9 and t9 >= t0) else 0.0
    overlap = bool(t7 and tracker.t8_grounded_gen_end and t7 < tracker.t8_grounded_gen_end)

    return BenchmarkTurnRecord(
        env="LIVE",
        mode="STREAMING" if streaming else "BASELINE",
        query_id=qid,
        query_label=qlabel,
        iteration=iteration,
        user_utterance=utterance,
        response_text=state.history[-1].content if state.history else "",
        t0_turn_start=t0,
        t1_decision_start=tracker.t1_decision_start,
        t2_decision_end=tracker.t2_decision_end,
        t3_retrieval_start=tracker.t3_retrieval_start,
        t4_retrieval_end=tracker.t4_retrieval_end,
        t5_grounded_synth_start=tracker.t5_grounded_synth_start,
        t6_first_grounded_phrase=tracker.t6_first_grounded_phrase,
        t7_first_audio_sent=t7,
        t8_grounded_gen_end=tracker.t8_grounded_gen_end,
        t9_playback_complete=t9,
        ttfa_ms=ttfa,
        time_to_first_grounded_phrase_ms=phrase_lat,
        tts_latency_ms=tts_lat,
        total_response_time_ms=total_time,
        streaming_overlap=overlap,
    )


# =============================================================================
# Statistical Helpers
# =============================================================================

def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return round(s[f] + (s[c] - s[f]) * d, 2)


def summarize_records(records: List[BenchmarkTurnRecord]) -> Dict[str, Any]:
    ttfas = [r.ttfa_ms for r in records if r.ttfa_ms > 0]
    first_phrases = [r.time_to_first_grounded_phrase_ms for r in records if r.time_to_first_grounded_phrase_ms > 0]
    tts_lats = [r.tts_latency_ms for r in records if r.tts_latency_ms > 0]
    totals = [r.total_response_time_ms for r in records if r.total_response_time_ms > 0]
    overlap_count = sum(1 for r in records if r.streaming_overlap)
    total_count = len(records)

    return {
        "count": total_count,
        "ttfa_p50": percentile(ttfas, 50),
        "ttfa_p95": percentile(ttfas, 95),
        "ttfa_min": min(ttfas) if ttfas else 0.0,
        "ttfa_max": max(ttfas) if ttfas else 0.0,
        "first_phrase_p50": percentile(first_phrases, 50),
        "first_phrase_p95": percentile(first_phrases, 95),
        "tts_lat_p50": percentile(tts_lats, 50),
        "tts_lat_p95": percentile(tts_lats, 95),
        "total_time_p50": percentile(totals, 50),
        "total_time_p95": percentile(totals, 95),
        "overlap_pct": round((overlap_count / total_count * 100.0), 1) if total_count else 0.0,
    }


def write_csv_artifact(records: List[BenchmarkTurnRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "env",
        "mode",
        "query_id",
        "query_label",
        "iteration",
        "user_utterance",
        "response_text",
        "t0_turn_start",
        "t1_decision_start",
        "t2_decision_end",
        "t3_retrieval_start",
        "t4_retrieval_end",
        "t5_grounded_synth_start",
        "t6_first_grounded_phrase",
        "t7_first_audio_sent",
        "t8_grounded_gen_end",
        "t9_playback_complete",
        "ttfa_ms",
        "time_to_first_grounded_phrase_ms",
        "tts_latency_ms",
        "total_response_time_ms",
        "streaming_overlap",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


# =============================================================================
# Main Orchestrator
# =============================================================================

async def main():
    print("=" * 80)
    print("  PHASE 5C -- PHASE 4: END-TO-END VOICE LATENCY BENCHMARK")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python:    {sys.version.split()[0]}")

    settings = get_settings()
    all_records: List[BenchmarkTurnRecord] = []

    # -------------------------------------------------------------------------
    # PART 1: MOCK BENCHMARK (10 iterations x 5 queries x 2 modes = 100 turns)
    # -------------------------------------------------------------------------
    mock_iterations = 10
    print("\n" + "-" * 80)
    print(f"[1/2] RUNNING MOCK BENCHMARK ({mock_iterations} iterations x 5 queries x 2 modes = 100 turns)...")
    print("-" * 80)

    for q in BENCHMARK_QUERIES:
        qid = q["id"]
        qlabel = q["label"]
        print(f"\n>> Query {qid}: [{qlabel}] \"{q['text']}\"")

        # Streaming
        for i in range(1, mock_iterations + 1):
            rec = await run_single_mock_turn(q, iteration=i, streaming=True)
            all_records.append(rec)
        print(f"   [OK] Streaming ({mock_iterations} runs) completed")

        # Baseline
        for i in range(1, mock_iterations + 1):
            rec = await run_single_mock_turn(q, iteration=i, streaming=False)
            all_records.append(rec)
        print(f"   [OK] Baseline  ({mock_iterations} runs) completed")

    print("\nMock Benchmark successfully completed.")

    # -------------------------------------------------------------------------
    # PART 2: LIVE BENCHMARK (Live Gemini & Live Deepgram Flux TTS)
    # -------------------------------------------------------------------------
    gemini_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    deepgram_key = settings.deepgram_api_key.get_secret_value() if settings.deepgram_api_key else ""
    can_run_live = bool(gemini_key and len(gemini_key) > 10 and deepgram_key and len(deepgram_key) > 10)

    live_records: List[BenchmarkTurnRecord] = []
    if can_run_live:
        live_iterations = 3
        total_live_turns = live_iterations * len(BENCHMARK_QUERIES) * 2
        print("\n" + "-" * 80)
        print(f"[2/2] RUNNING LIVE BENCHMARK ({live_iterations} iterations x 5 queries x 2 modes = {total_live_turns} turns)...")
        print(f"      Live Model: {settings.gemini_model} | Live TTS: aura-asteria-en")
        print("-" * 80)

        # Initialize live providers
        llm = GeminiLLMProvider(settings=settings)
        kp = GroundedKnowledgeProvider(
            embedding_provider=get_embedding_provider(settings=settings, force_mock=False),
            in_memory=True,
        )
        ingestion = KnowledgeIngestionPipeline(knowledge_provider=kp)
        await ingestion.ingest_domain(DomainType.INFRA_BUILDER)
        if hasattr(kp, "warmup"):
            kp.warmup()

        engine = ConversationEngine(llm_provider=llm, knowledge_provider=kp, tool_provider=MockToolProvider())
        state_mgr = ConversationStateManager(settings=settings)

        for q in BENCHMARK_QUERIES:
            qid = q["id"]
            qlabel = q["label"]
            print(f"\n>> LIVE Query {qid}: [{qlabel}] \"{q['text']}\"")

            # Streaming
            for i in range(1, live_iterations + 1):
                try:
                    rec = await run_single_live_turn(q, iteration=i, streaming=True, engine=engine, state_mgr=state_mgr, settings=settings)
                    all_records.append(rec)
                    live_records.append(rec)
                    print(f"   [Live Stream run {i}] TTFA: {rec.ttfa_ms:.1f}ms | 1stPhrase: {rec.time_to_first_grounded_phrase_ms:.1f}ms | Overlap: {rec.streaming_overlap}")
                except Exception as exc:
                    print(f"   [Live Stream run {i} ERROR]: {exc}")
                await asyncio.sleep(1.0)  # Rate pacing

            # Baseline
            for i in range(1, live_iterations + 1):
                try:
                    rec = await run_single_live_turn(q, iteration=i, streaming=False, engine=engine, state_mgr=state_mgr, settings=settings)
                    all_records.append(rec)
                    live_records.append(rec)
                    print(f"   [Live Baseline run {i}] TTFA: {rec.ttfa_ms:.1f}ms | Total: {rec.total_response_time_ms:.1f}ms")
                except Exception as exc:
                    print(f"   [Live Baseline run {i} ERROR]: {exc}")
                await asyncio.sleep(1.0)  # Rate pacing

        try:
            await llm.aclose()
        except Exception:
            pass
    else:
        print("\n[2/2] LIVE BENCHMARK SKIPPED: Missing valid GEMINI_API_KEY or DEEPGRAM_API_KEY.")

    # -------------------------------------------------------------------------
    # PART 3: SAVE CSV ARTIFACTS
    # -------------------------------------------------------------------------
    csv_primary = PROJECT_ROOT / "benchmarks" / "results" / "benchmark_e2e_latency.csv"
    write_csv_artifact(all_records, csv_primary)
    print(f"\n[OK] Saved CSV artifact to: {csv_primary}")

    # Also save copy to conversation artifact directory if available
    brain_dir = Path("C:/Users/Pawanganesh/.gemini/antigravity/brain/8c186d8c-bcba-41f0-9867-dd2a2c12297b")
    if brain_dir.exists():
        csv_brain = brain_dir / "benchmark_results.csv"
        write_csv_artifact(all_records, csv_brain)
        print(f"[OK] Saved brain artifact copy to: {csv_brain}")

    # -------------------------------------------------------------------------
    # PART 4: STATISTICAL REPORT SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  LATENCY BENCHMARK STATISTICAL SUMMARY")
    print("=" * 80)

    # Mock Summary
    mock_stream = [r for r in all_records if r.env == "MOCK" and r.mode == "STREAMING"]
    mock_base = [r for r in all_records if r.env == "MOCK" and r.mode == "BASELINE"]
    ms_sum = summarize_records(mock_stream)
    mb_sum = summarize_records(mock_base)

    print("\n--- MOCK BENCHMARK SUMMARY (N=50 streaming, N=50 baseline) ---")
    print(f"{'Metric':<32} | {'Streaming':<18} | {'Baseline':<18} | {'Delta / Reduction':<18}")
    print("-" * 92)
    ttfa_delta = ms_sum["ttfa_p50"] - mb_sum["ttfa_p50"]
    ttfa_pct = ((ms_sum["ttfa_p50"] - mb_sum["ttfa_p50"]) / mb_sum["ttfa_p50"] * 100.0) if mb_sum["ttfa_p50"] else 0.0
    print(f"{'TTFA p50':<32} | {ms_sum['ttfa_p50']:>10.2f} ms     | {mb_sum['ttfa_p50']:>10.2f} ms     | {ttfa_delta:>+8.2f}ms ({ttfa_pct:.1f}%)")
    print(f"{'TTFA p95':<32} | {ms_sum['ttfa_p95']:>10.2f} ms     | {mb_sum['ttfa_p95']:>10.2f} ms     |")
    print(f"{'Time to 1st Grounded Phrase p50':<32} | {ms_sum['first_phrase_p50']:>10.2f} ms     | {mb_sum['first_phrase_p50']:>10.2f} ms     |")
    print(f"{'Total Response Time p50':<32} | {ms_sum['total_time_p50']:>10.2f} ms     | {mb_sum['total_time_p50']:>10.2f} ms     |")
    print(f"{'Streaming Overlap (T7 < T8)':<32} | {ms_sum['overlap_pct']:>10.1f} %      | {mb_sum['overlap_pct']:>10.1f} %      |")

    # Live Summary
    if live_records:
        live_stream = [r for r in all_records if r.env == "LIVE" and r.mode == "STREAMING"]
        live_base = [r for r in all_records if r.env == "LIVE" and r.mode == "BASELINE"]
        ls_sum = summarize_records(live_stream)
        lb_sum = summarize_records(live_base)

        print(f"\n--- LIVE BENCHMARK SUMMARY (N={len(live_stream)} streaming, N={len(live_base)} baseline) ---")
        print(f"{'Metric':<32} | {'Streaming':<18} | {'Baseline':<18} | {'Delta / Reduction':<18}")
        print("-" * 92)
        l_ttfa_delta = ls_sum["ttfa_p50"] - lb_sum["ttfa_p50"]
        l_ttfa_pct = ((ls_sum["ttfa_p50"] - lb_sum["ttfa_p50"]) / lb_sum["ttfa_p50"] * 100.0) if lb_sum["ttfa_p50"] else 0.0
        print(f"{'TTFA p50':<32} | {ls_sum['ttfa_p50']:>10.2f} ms     | {lb_sum['ttfa_p50']:>10.2f} ms     | {l_ttfa_delta:>+8.2f}ms ({l_ttfa_pct:.1f}%)")
        print(f"{'TTFA p95':<32} | {ls_sum['ttfa_p95']:>10.2f} ms     | {lb_sum['ttfa_p95']:>10.2f} ms     |")
        print(f"{'Time to 1st Grounded Phrase p50':<32} | {ls_sum['first_phrase_p50']:>10.2f} ms     | {lb_sum['first_phrase_p50']:>10.2f} ms     |")
        print(f"{'TTS Latency p50':<32} | {ls_sum['tts_lat_p50']:>10.2f} ms     | {lb_sum['tts_lat_p50']:>10.2f} ms     |")
        print(f"{'Total Response Time p50':<32} | {ls_sum['total_time_p50']:>10.2f} ms     | {lb_sum['total_time_p50']:>10.2f} ms     |")
        print(f"{'Streaming Overlap (T7 < T8)':<32} | {ls_sum['overlap_pct']:>10.1f} %      | {lb_sum['overlap_pct']:>10.1f} %      |")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
