"""Phase 16 Latency Benchmark Runner.

Executes real/mock turns through ConversationEngine, STTService, DeepgramFluxTTS,
and measures exact monotonic timestamps using TurnLatencyTracker:
1. Normal conversational response
2. RAG response (Vayvora)
3. RAG response (EduSaaS)
4. Tool action (Calendar booking)
5. Tool action (Email sending)
6. Tool action (Lead capture / CRM)
7. RAG + Tool action
"""

import asyncio
import time
from typing import Any, Dict

from src.config import get_settings
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.llm import GeminiLLMProvider
from src.core.types import CallDirection, DomainType
from src.rag.retriever import GroundedKnowledgeProvider
from src.rag.embeddings import MockEmbeddingProvider
from src.state.manager import ConversationStateManager
from src.tools.mcp_client import MockToolProvider
from src.voice.latency_tracker import TurnLatencyTracker
from src.voice.stt.stt_service import STTService
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS


async def run_benchmark():
    settings = get_settings()
    llm = GeminiLLMProvider(settings=settings)
    stt = STTService()
    tts = DeepgramFluxTTS()
    tools = MockToolProvider()
    rag = GroundedKnowledgeProvider(embedding_provider=MockEmbeddingProvider(dimension=64), in_memory=True)
    engine = ConversationEngine(llm_provider=llm, knowledge_provider=rag, tool_provider=tools)
    mgr = ConversationStateManager(settings=settings)

    print("=" * 60)
    print("VAYVORA AI VOICE AGENT — LATENCY BENCHMARK (PHASE 16)")
    print("=" * 60)

    # 1. Warm-up LLM and TTS connections
    print("\nWarming up connections (TLS handshake & HTTP keep-alive)...")
    await tts.connect()
    dummy_tracker = TurnLatencyTracker("warmup", 0)
    warmup_state = mgr.create_inbound_state(call_id="call-warmup", caller_phone="+15551234567", domain=DomainType.GENERAL)
    await engine.process_user_turn(warmup_state, "Hello", tracker=dummy_tracker)
    async for _ in tts.synthesize_stream("Ready."):
        pass
    print("Warm-up complete.\n")

    scenarios = [
        ("1. Normal Conversational Turn", DomainType.GENERAL, "Hi, can you hear me?"),
        ("2. RAG Response (Vayvora)", DomainType.VAYVORA, "What AI calling features and telephony services does Vayvora provide?"),
        ("3. RAG Response (EduSaaS)", DomainType.EDUSAAS, "Tell me about the Data Science curriculum and course fees."),
        ("4. Tool Action (Calendar Booking)", DomainType.VAYVORA, "Please schedule a consultation for tomorrow at 3pm"),
        ("5. Tool Action (Email Sending)", DomainType.EDUSAAS, "Please send me the brochure to test@example.com"),
        ("6. Tool Action (Lead Capture / CRM)", DomainType.VAYVORA, "My name is John and I would like to register my interest"),
        ("7. RAG + Tool Action", DomainType.EDUSAAS, "What are the admission requirements for AI and send the details to test@example.com"),
    ]

    results = []

    for label, domain, utterance in scenarios:
        tracker = TurnLatencyTracker("turn_test", 1)
        tracker.mark_speech_start()
        # Simulate short audio capture
        await asyncio.sleep(0.01)
        tracker.mark_speech_end()

        call_id = f"call-bench-{int(time.time()*1000)}"
        state = mgr.create_inbound_state(
            call_id=call_id,
            caller_phone="+15551234567",
            domain=domain,
            domain_locked=True,
        )
        if "test@example.com" in utterance:
            state.caller.email = "test@example.com"

        # Engine turn
        turn_result = await engine.process_user_turn(state, utterance, tracker=tracker)
        response_text = turn_result.response_text

        # TTS Stream
        tracker.mark_tts_start()
        first_audio_received = False
        async for chunk in tts.synthesize_stream(response_text[:120]):
            if not first_audio_received:
                tracker.mark_tts_first_audio()
                tracker.mark_first_audio_sent()
                first_audio_received = True
                break

        summary = tracker.to_dict()
        results.append((label, summary, response_text[:60]))

        print(f"Scenario: {label}")
        print(f"  Utterance: \"{utterance}\"")
        print(f"  Response:  \"{response_text[:60]}...\"")
        print(f"  Reused 1st Gemini Decision: {summary['reused_first_response']}")
        print(f"  Decision Latency (LLM):     {summary['gemini_latency_ms']:.1f} ms")
        if summary['rag_latency_ms']:
            print(f"  RAG Latency:                {summary['rag_latency_ms']:.1f} ms")
        if summary['tool_latency_ms']:
            print(f"  Tool Execution Latency:     {summary['tool_latency_ms']:.1f} ms")
        print(f"  TTS First Audio Latency:    {summary['tts_first_audio_latency_ms']:.1f} ms")
        print(f"  TOTAL Time-to-First-Audio:  {summary['speech_end_to_first_audio_ms']:.1f} ms ({(summary['speech_end_to_first_audio_ms']/1000):.3f}s)")
        print("-" * 60)

    # Cleanup
    await tts.close()
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(run_benchmark())
