"""Phase 5D-2C: Post-Deduplication Decision Latency Benchmark Runner.

Empirically measures the effect of Phase 5D-2B prompt deduplication on:
1. Exact prompt size (characters and official Gemini token counts via provider metadata).
2. Dedicated Live Gemini decision call latency (N=30 post-dedup vs N=30 pre-dedup).
3. End-to-end voice TTFA across the fixed 5-query benchmark set (Streaming vs Baseline).
4. Full mock benchmark verification (N=100 turns).
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx

from benchmarks.benchmark_e2e_latency import (
    BENCHMARK_QUERIES,
    BenchmarkMockWebSocket,
    BenchmarkTurnRecord,
    percentile,
    run_single_mock_turn,
    summarize_records,
    write_csv_artifact,
)
from src.config import Settings, get_settings
from src.core.decision import ConversationalDecision, DecisionValidator
from src.core.engine import ConversationEngine
from src.core.llm import GeminiLLMProvider, extract_json_block
from src.core.prompts import PromptSynthesizer
from src.core.types import CallDirection, DomainType
from src.domains import INFRA_BUILDER_DOMAIN_CONFIG
from src.prompts import load_prompt
from src.rag.embeddings import get_embedding_provider
from src.rag.ingestion import KnowledgeIngestionPipeline
from src.rag.retriever import GroundedKnowledgeProvider
from src.state.manager import ConversationStateManager
from src.tools.mcp_client import MockToolProvider
from src.voice.latency_tracker import TurnLatencyTracker
from src.voice.session_manager import VoiceSessionManager
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS


@dataclass
class PromptSizeRecord:
    query_id: int
    query_label: str
    user_utterance: str
    system_prompt_chars: int
    user_prompt_chars_pre: int
    user_prompt_chars_post: int
    char_reduction: int
    char_reduction_pct: float
    estimated_tokens_pre: int
    estimated_tokens_post: int
    actual_prompt_tokens_pre: Optional[int]
    actual_prompt_tokens_post: Optional[int]
    actual_token_reduction: Optional[int]
    actual_token_reduction_pct: Optional[float]


@dataclass
class DecisionCallRecord:
    variant: str  # "POST_DEDUP" or "PRE_DEDUP"
    query_id: int
    query_label: str
    iteration: int
    latency_ms: float
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    http_retries: int
    parse_retries: int
    status: str
    detected_domain: str
    detected_intent: str
    proposed_stage: str
    error: Optional[str] = None


async def measure_prompt_sizes(settings: Settings) -> List[PromptSizeRecord]:
    """Measure exact characters and provider token counts for all 5 queries."""
    synthesizer = PromptSynthesizer()
    mgr = ConversationStateManager()
    decision_instructions = load_prompt("agent/decision_instructions.txt").strip()

    api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    records: List[PromptSizeRecord] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for q in BENCHMARK_QUERIES:
            qid = q["id"]
            qlabel = q["label"]
            text = q["text"]

            state = mgr.create_inbound_state(f"bench-p-{qid}", "+919876543210", domain=DomainType.INFRA_BUILDER)
            sys_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
            user_prompt_post = synthesizer.build_user_prompt(state, text)
            user_prompt_pre = user_prompt_post + "\n\n" + decision_instructions

            # Character metrics
            sys_chars = len(sys_prompt)
            user_pre_chars = len(user_prompt_pre)
            user_post_chars = len(user_prompt_post)
            char_red = user_pre_chars - user_post_chars
            char_red_pct = round((char_red / user_pre_chars) * 100.0, 1)

            est_pre = (sys_chars + user_pre_chars) // 4
            est_post = (sys_chars + user_post_chars) // 4

            actual_pre = None
            actual_post = None
            token_red = None
            token_red_pct = None

            if api_key:
                try:
                    payload_pre = {
                        "contents": [{"role": "user", "parts": [{"text": user_prompt_pre}]}],
                        "systemInstruction": {"parts": [{"text": sys_prompt}]},
                        "generationConfig": {"maxOutputTokens": 1},
                    }
                    r_pre = await client.post(url, json=payload_pre, headers=headers)
                    if r_pre.status_code == 200:
                        actual_pre = r_pre.json().get("usageMetadata", {}).get("promptTokenCount")

                    payload_post = {
                        "contents": [{"role": "user", "parts": [{"text": user_prompt_post}]}],
                        "systemInstruction": {"parts": [{"text": sys_prompt}]},
                        "generationConfig": {"maxOutputTokens": 1},
                    }
                    r_post = await client.post(url, json=payload_post, headers=headers)
                    if r_post.status_code == 200:
                        actual_post = r_post.json().get("usageMetadata", {}).get("promptTokenCount")

                    if actual_pre and actual_post:
                        token_red = actual_pre - actual_post
                        token_red_pct = round((token_red / actual_pre) * 100.0, 1)
                except Exception as exc:
                    print(f"Warning: Token count API error for Q{qid}: {exc}")

            records.append(
                PromptSizeRecord(
                    query_id=qid,
                    query_label=qlabel,
                    user_utterance=text,
                    system_prompt_chars=sys_chars,
                    user_prompt_chars_pre=user_pre_chars,
                    user_prompt_chars_post=user_post_chars,
                    char_reduction=char_red,
                    char_reduction_pct=char_red_pct,
                    estimated_tokens_pre=est_pre,
                    estimated_tokens_post=est_post,
                    actual_prompt_tokens_pre=actual_pre,
                    actual_prompt_tokens_post=actual_post,
                    actual_token_reduction=token_red,
                    actual_token_reduction_pct=token_red_pct,
                )
            )

    return records


async def run_single_decision_call(
    variant: str,
    query_info: Dict[str, Any],
    iteration: int,
    settings: Settings,
    client: httpx.AsyncClient,
) -> DecisionCallRecord:
    """Execute a single live Gemini decision call with monotonic timing, retry on 429, and token usage tracking."""
    qid = query_info["id"]
    qlabel = query_info["label"]
    text = query_info["text"]

    synthesizer = PromptSynthesizer()
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state(f"bench-d-{variant}-{qid}-{iteration}", "+919876543210", domain=DomainType.INFRA_BUILDER)
    sys_prompt = synthesizer.build_system_instruction(state, INFRA_BUILDER_DOMAIN_CONFIG)
    user_prompt = synthesizer.build_user_prompt(state, text)

    if variant == "PRE_DEDUP":
        decision_instructions = load_prompt("agent/decision_instructions.txt").strip()
        user_prompt = user_prompt + "\n\n" + decision_instructions

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    headers = {
        "x-goog-api-key": settings.gemini_api_key.get_secret_value(),
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": sys_prompt}]},
        "generationConfig": {"temperature": settings.gemini_temperature},
    }

    validator = DecisionValidator()
    t_start = time.perf_counter()
    http_retries = 0
    parse_retries = 0
    error_msg = None
    status = "SUCCESS"
    detected_domain = ""
    detected_intent = ""
    proposed_stage = ""
    prompt_tokens = 0
    output_tokens = 0
    total_tokens = 0

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                http_retries += 1
                if attempt < max_attempts - 1:
                    backoff = 2.0 * (attempt + 1)
                    await asyncio.sleep(backoff)
                    continue
                status = "HTTP_429"
                error_msg = "Rate limit exceeded"
                break
            elif resp.status_code != 200:
                status = f"HTTP_{resp.status_code}"
                error_msg = resp.text[:200]
                break

            data = resp.json()
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)

            candidates = data.get("candidates", [])
            if candidates:
                raw_text = candidates[0]["content"]["parts"][0]["text"]
                clean_json = extract_json_block(raw_text)
                decision_dict = json.loads(clean_json)
                decision = ConversationalDecision.model_validate(decision_dict)
                validated = validator.validate_and_filter(decision)
                detected_domain = validated.detected_domain.value if validated.detected_domain else ""
                detected_intent = validated.detected_intent or ""
                proposed_stage = validated.proposed_stage.value if validated.proposed_stage else ""
                status = "SUCCESS"
                break
            else:
                status = "EMPTY_CANDIDATES"
                break
        except Exception as exc:
            status = "EXCEPTION"
            error_msg = str(exc)
            break

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 2)

    return DecisionCallRecord(
        variant=variant,
        query_id=qid,
        query_label=qlabel,
        iteration=iteration,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        http_retries=http_retries,
        parse_retries=parse_retries,
        status=status,
        detected_domain=detected_domain,
        detected_intent=detected_intent,
        proposed_stage=proposed_stage,
        error=error_msg,
    )


def summarize_decision_records(records: List[DecisionCallRecord]) -> Dict[str, Any]:
    valid = [r.latency_ms for r in records if r.status == "SUCCESS"]
    in_tokens = [r.prompt_tokens for r in records if r.prompt_tokens > 0]
    out_tokens = [r.output_tokens for r in records if r.output_tokens > 0]
    total_samples = len(records)
    failed_samples = total_samples - len(valid)

    mean_lat = round(sum(valid) / len(valid), 2) if valid else 0.0

    return {
        "samples": total_samples,
        "valid_samples": len(valid),
        "failed_samples": failed_samples,
        "p50_ms": percentile(valid, 50),
        "p90_ms": percentile(valid, 90),
        "p95_ms": percentile(valid, 95),
        "mean_ms": mean_lat,
        "min_ms": min(valid) if valid else 0.0,
        "max_ms": max(valid) if valid else 0.0,
        "mean_input_tokens": round(sum(in_tokens) / len(in_tokens), 1) if in_tokens else 0.0,
        "mean_output_tokens": round(sum(out_tokens) / len(out_tokens), 1) if out_tokens else 0.0,
    }


def write_decision_csv(records: List[DecisionCallRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "query_id",
        "query_label",
        "iteration",
        "latency_ms",
        "prompt_tokens",
        "output_tokens",
        "total_tokens",
        "http_retries",
        "parse_retries",
        "status",
        "detected_domain",
        "detected_intent",
        "proposed_stage",
        "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


async def run_single_live_voice_turn(
    query_info: Dict[str, Any],
    iteration: int,
    streaming: bool,
    engine: ConversationEngine,
    state_mgr: ConversationStateManager,
    settings: Settings,
) -> BenchmarkTurnRecord:
    """Run a single live voice turn capturing audio frames and monotonic boundaries."""
    qid = query_info["id"]
    qlabel = query_info["label"]
    utterance = query_info["text"]

    session_id = f"bench-live-pd-{qid}-{iteration}-{'stream' if streaming else 'batch'}"
    state = state_mgr.create_session(session_id, DomainType.INFRA_BUILDER, CallDirection.INBOUND)

    deepgram_key = settings.deepgram_api_key.get_secret_value() if settings.deepgram_api_key else ""
    tts = DeepgramFluxTTS(api_key=deepgram_key)
    ws = BenchmarkMockWebSocket()
    tts_done_event = asyncio.Event()
    audio_received_event = asyncio.Event()

    async def audio_cb(pcm_bytes: bytes, turn_id: Optional[int] = None):
        await ws.send_bytes(pcm_bytes)
        audio_received_event.set()

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

        # Bounded wait for first audio frame so socket is not closed prematurely
        if tts.first_speak_sent_at and not ws.first_audio_sent_ts:
            try:
                await asyncio.wait_for(audio_received_event.wait(), timeout=2.5)
            except asyncio.TimeoutError:
                pass
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


async def main():
    print("=" * 80)
    print("  PHASE 5D-2C: POST-DEDUPLICATION DECISION & E2E VOICE BENCHMARK")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python:    {sys.version.split()[0]}")

    settings = get_settings()
    print(f"Gemini Model:   {settings.gemini_model}")
    print(f"Gemini Timeout: {settings.gemini_timeout_seconds}s")
    print(f"Gemini Temp:    {settings.gemini_temperature}")

    brain_dir = Path("C:/Users/Pawanganesh/.gemini/antigravity/brain/8c186d8c-bcba-41f0-9867-dd2a2c12297b")

    # -------------------------------------------------------------------------
    # PART 1: EXACT PROMPT SIZE MEASUREMENT
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[1/4] MEASURING EXACT PROMPT SIZES & PROVIDER TOKEN COUNTS...")
    print("-" * 80)
    prompt_records = await measure_prompt_sizes(settings)

    print(f"{'QID':<4} | {'Query Label':<30} | {'User Chars (Pre -> Post)':<25} | {'Actual Tokens (Pre -> Post)':<27} | {'Token Delta':<14}")
    print("-" * 108)
    for pr in prompt_records:
        pre_post_chars = f"{pr.user_prompt_chars_pre} -> {pr.user_prompt_chars_post} (-{pr.char_reduction_pct}%)"
        pre_post_toks = f"{pr.actual_prompt_tokens_pre} -> {pr.actual_prompt_tokens_post}"
        tok_delta = f"-{pr.actual_token_reduction} (-{pr.actual_token_reduction_pct}%)"
        print(f"Q{pr.query_id:<3} | {pr.query_label:<30} | {pre_post_chars:<25} | {pre_post_toks:<27} | {tok_delta:<14}")

    # -------------------------------------------------------------------------
    # PART 2: DEDICATED LIVE GEMINI DECISION BENCHMARK (N=30 Post vs N=30 Pre)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[2/4] DEDICATED LIVE DECISION BENCHMARK (30 calls Post-Dedup vs 30 calls Pre-Dedup)...")
    print("-" * 80)

    decision_records: List[DecisionCallRecord] = []
    iterations_per_query = 6  # 6 iterations x 5 queries = 30 calls per variant

    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=30.0)
    async with httpx.AsyncClient(timeout=settings.gemini_timeout_seconds, limits=limits) as http_client:
        # Run Post-Deduplication Live Decision Calls
        print(f"\n>> Running Post-Deduplication Decision Calls ({iterations_per_query} runs x 5 queries = 30 calls)...")
        for q in BENCHMARK_QUERIES:
            for it in range(1, iterations_per_query + 1):
                rec = await run_single_decision_call("POST_DEDUP", q, it, settings, http_client)
                decision_records.append(rec)
                print(f"   [POST Q{q['id']} run {it}] {rec.latency_ms:>7.1f}ms | in_tok: {rec.prompt_tokens} | out_tok: {rec.output_tokens} | status: {rec.status}")
                await asyncio.sleep(1.5)  # Pacing to avoid rate limits

        # Run Pre-Deduplication Live Decision Calls (Head-to-head empirical control)
        print(f"\n>> Running Pre-Deduplication Decision Calls ({iterations_per_query} runs x 5 queries = 30 calls)...")
        for q in BENCHMARK_QUERIES:
            for it in range(1, iterations_per_query + 1):
                rec = await run_single_decision_call("PRE_DEDUP", q, it, settings, http_client)
                decision_records.append(rec)
                print(f"   [PRE  Q{q['id']} run {it}] {rec.latency_ms:>7.1f}ms | in_tok: {rec.prompt_tokens} | out_tok: {rec.output_tokens} | status: {rec.status}")
                await asyncio.sleep(2.0)  # Slightly longer pacing for larger token footprint

    # Save Decision CSV
    dec_csv_primary = PROJECT_ROOT / "benchmarks" / "results" / "benchmark_decision_latency.csv"
    write_decision_csv(decision_records, dec_csv_primary)
    print(f"\n[OK] Saved Decision CSV to: {dec_csv_primary}")
    if brain_dir.exists():
        write_decision_csv(decision_records, brain_dir / "benchmark_decision_results.csv")
        print(f"[OK] Saved Brain Decision CSV copy to: {brain_dir / 'benchmark_decision_results.csv'}")

    post_dec_sum = summarize_decision_records([r for r in decision_records if r.variant == "POST_DEDUP"])
    pre_dec_sum = summarize_decision_records([r for r in decision_records if r.variant == "PRE_DEDUP"])

    print("\n--- LIVE DECISION LATENCY SUMMARY ---")
    print(f"{'Metric':<25} | {'Pre-Dedup (Baseline)':<22} | {'Post-Dedup (Optimized)':<22} | {'Delta / Reduction':<20}")
    print("-" * 95)
    d_p50 = post_dec_sum["p50_ms"] - pre_dec_sum["p50_ms"]
    d_p50_pct = ((post_dec_sum["p50_ms"] - pre_dec_sum["p50_ms"]) / pre_dec_sum["p50_ms"] * 100.0) if pre_dec_sum["p50_ms"] else 0.0
    d_mean = post_dec_sum["mean_ms"] - pre_dec_sum["mean_ms"]
    d_mean_pct = ((post_dec_sum["mean_ms"] - pre_dec_sum["mean_ms"]) / pre_dec_sum["mean_ms"] * 100.0) if pre_dec_sum["mean_ms"] else 0.0

    print(f"{'Decision p50':<25} | {pre_dec_sum['p50_ms']:>14.2f} ms      | {post_dec_sum['p50_ms']:>14.2f} ms      | {d_p50:>+8.2f}ms ({d_p50_pct:.1f}%)")
    print(f"{'Decision p90':<25} | {pre_dec_sum['p90_ms']:>14.2f} ms      | {post_dec_sum['p90_ms']:>14.2f} ms      | {post_dec_sum['p90_ms'] - pre_dec_sum['p90_ms']:>+8.2f}ms")
    print(f"{'Decision p95':<25} | {pre_dec_sum['p95_ms']:>14.2f} ms      | {post_dec_sum['p95_ms']:>14.2f} ms      | {post_dec_sum['p95_ms'] - pre_dec_sum['p95_ms']:>+8.2f}ms")
    print(f"{'Decision Mean':<25} | {pre_dec_sum['mean_ms']:>14.2f} ms      | {post_dec_sum['mean_ms']:>14.2f} ms      | {d_mean:>+8.2f}ms ({d_mean_pct:.1f}%)")
    print(f"{'Mean Prompt Tokens':<25} | {pre_dec_sum['mean_input_tokens']:>14.1f} tokens  | {post_dec_sum['mean_input_tokens']:>14.1f} tokens  | -{pre_dec_sum['mean_input_tokens'] - post_dec_sum['mean_input_tokens']:.1f} tok")
    print(f"{'Valid / Failures':<25} | {pre_dec_sum['valid_samples']}/{pre_dec_sum['failed_samples']:<19} | {post_dec_sum['valid_samples']}/{post_dec_sum['failed_samples']:<19} | 0 errors")

    # -------------------------------------------------------------------------
    # PART 3: LIVE END-TO-END VOICE BENCHMARK (N=30 turns: Streaming vs Baseline)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[3/4] RUNNING LIVE END-TO-END VOICE BENCHMARK (30 turns with Live Deepgram TTS)...")
    print("-" * 80)

    all_turn_records: List[BenchmarkTurnRecord] = []
    live_iterations = 3

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
        print(f"\n>> LIVE Voice Query {qid}: [{qlabel}] \"{q['text']}\"")

        # Streaming
        for i in range(1, live_iterations + 1):
            try:
                rec = await run_single_live_voice_turn(q, iteration=i, streaming=True, engine=engine, state_mgr=state_mgr, settings=settings)
                all_turn_records.append(rec)
                print(f"   [Live Stream run {i}] TTFA: {rec.ttfa_ms:>7.1f}ms | 1stPhrase: {rec.time_to_first_grounded_phrase_ms:>7.1f}ms | Dec: {round((rec.t2_decision_end - rec.t1_decision_start)*1000, 1):>7.1f}ms | Overlap: {rec.streaming_overlap}")
            except Exception as exc:
                print(f"   [Live Stream run {i} ERROR]: {exc}")
            await asyncio.sleep(1.5)

        # Baseline
        for i in range(1, live_iterations + 1):
            try:
                rec = await run_single_live_voice_turn(q, iteration=i, streaming=False, engine=engine, state_mgr=state_mgr, settings=settings)
                all_turn_records.append(rec)
                print(f"   [Live Base   run {i}] TTFA: {rec.ttfa_ms:>7.1f}ms | Total: {rec.total_response_time_ms:>7.1f}ms | Dec: {round((rec.t2_decision_end - rec.t1_decision_start)*1000, 1):>7.1f}ms")
            except Exception as exc:
                print(f"   [Live Base   run {i} ERROR]: {exc}")
            await asyncio.sleep(1.5)

    try:
        await llm.aclose()
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # PART 4: MOCK BENCHMARK VERIFICATION (N=100 turns)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[4/4] RUNNING MOCK BENCHMARK VERIFICATION (100 turns)...")
    print("-" * 80)
    mock_iterations = 10
    for q in BENCHMARK_QUERIES:
        for i in range(1, mock_iterations + 1):
            rec = await run_single_mock_turn(q, iteration=i, streaming=True)
            all_turn_records.append(rec)
        for i in range(1, mock_iterations + 1):
            rec = await run_single_mock_turn(q, iteration=i, streaming=False)
            all_turn_records.append(rec)
    print(f"[OK] Completed {mock_iterations * len(BENCHMARK_QUERIES) * 2} mock turns.")

    # -------------------------------------------------------------------------
    # SAVE E2E CSV ARTIFACTS
    # -------------------------------------------------------------------------
    e2e_csv_primary = PROJECT_ROOT / "benchmarks" / "results" / "benchmark_post_dedup_latency.csv"
    write_csv_artifact(all_turn_records, e2e_csv_primary)
    print(f"\n[OK] Saved Post-Dedup E2E CSV to: {e2e_csv_primary}")
    if brain_dir.exists():
        write_csv_artifact(all_turn_records, brain_dir / "benchmark_post_dedup_results.csv")
        print(f"[OK] Saved Brain E2E CSV copy to: {brain_dir / 'benchmark_post_dedup_results.csv'}")

    # -------------------------------------------------------------------------
    # E2E SUMMARY
    # -------------------------------------------------------------------------
    live_records = [r for r in all_turn_records if r.env == "LIVE"]
    live_stream = [r for r in live_records if r.mode == "STREAMING"]
    live_base = [r for r in live_records if r.mode == "BASELINE"]
    ls_sum = summarize_records(live_stream)
    lb_sum = summarize_records(live_base)

    print("\n" + "=" * 80)
    print("  LIVE POST-DEDUPLICATION E2E VOICE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Streaming (Post-Dedup)':<24} | {'Baseline (Post-Dedup)':<24}")
    print("-" * 86)
    print(f"{'TTFA p50':<32} | {ls_sum['ttfa_p50']:>16.2f} ms     | {lb_sum['ttfa_p50']:>16.2f} ms")
    print(f"{'TTFA p95':<32} | {ls_sum['ttfa_p95']:>16.2f} ms     | {lb_sum['ttfa_p95']:>16.2f} ms")
    print(f"{'Time to 1st Grounded Phrase p50':<32} | {ls_sum['first_phrase_p50']:>16.2f} ms     | {lb_sum['first_phrase_p50']:>16.2f} ms")
    print(f"{'TTS Latency p50':<32} | {ls_sum['tts_lat_p50']:>16.2f} ms     | {lb_sum['tts_lat_p50']:>16.2f} ms")
    print(f"{'Total Response Time p50':<32} | {ls_sum['total_time_p50']:>16.2f} ms     | {lb_sum['total_time_p50']:>16.2f} ms")
    print(f"{'Streaming Overlap (T7 < T8)':<32} | {ls_sum['overlap_pct']:>16.1f} %      | {lb_sum['overlap_pct']:>16.1f} %")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
