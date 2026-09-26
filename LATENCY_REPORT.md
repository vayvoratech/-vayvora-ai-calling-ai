# Turn Latency & Telemetry Analysis Report

**Date**: September 26, 2026  
**System**: Unified Real-Time Conversational AI Voice Agent  
**Environment**: Windows 11, Python 3.11.9, FastAPI Workbench  
**Primary LLM Model**: `gemini-3.5-flash-lite` (with dynamic fallback)  
**Instrumentation**: Nanosecond-resolution stage timers in `src/core/engine.py`  

---

## 1. Executive Summary

Voice agent responsiveness is governed by conversational latency—the total duration between the end of caller speech and the first byte of synthesized agent audio. In telephony applications, target response times are under **800ms–1,200ms** for natural back-and-forth dialogue.

This report documents empirical latency measurements recorded across conversational turns during the comprehensive scenario audit, isolates system bottlenecks, and outlines concrete production optimizations.

---

## 2. Granular Stage Latency Breakdown

Each conversational turn executed through `ConversationEngine.process_user_turn` is instrumented into 5 distinct sequential stages:

```
[User Input Received]
        │
        ▼ (Stage 1: Prompt Construction)
[Dynamic Prompt Synthesizer]
        │
        ▼ (Stage 2: LLM Decision & Intent Routing)
[Gemini 3.5 Flash-Lite API]
        │
        ├──────────────────────────────────────┐
        ▼ (Stage 3: RAG Retrieval - if needed)  ▼ (Stage 5: Tool Execution - if proposed)
[Vector Store + BM25 + RRF]             [MCP Server / SMTP Mailer]
        │                                      │
        ▼ (Stage 4: Grounded Synthesis)        │
[Gemini Grounded Generation]                   │
        │                                      │
        └───────────────────┬──────────────────┘
                            ▼
                  [Spoken Response Out]
```

### Turn Timing Summary Across Audit Scenarios

| Turn Type / Description | Prompt Prep | Gemini Decision | RAG Retrieval | Grounded Gen | Tool Execution | Total Turn Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Pure Greeting** (Inbound Turn 1 "Hello") | 0.16 ms | 2,610.0 ms | — | — | — | **2,610.2 ms** |
| **Short Confirmation** (Outbound Turn 1 "yes") | 0.12 ms | 2,016.7 ms | — | — | — | **2,016.8 ms** |
| **Objection Handling** (Outbound Turn 1 "I'm busy") | 0.08 ms | 1,939.4 ms | — | — | — | **1,939.6 ms** |
| **RAG Retrieval - Warm** (EduSaaS Course Inquiry) | 0.08 ms | 2,167.3 ms | 21.5 ms | 1,879.7 ms | — | **4,068.7 ms** |
| **RAG Retrieval - Office Query** (General Inquiry) | 0.08 ms | 2,522.5 ms | 24.5 ms | 2,771.7 ms | — | **5,318.9 ms** |
| **MCP Tool Action** (Calendar Slot Search) | 0.11 ms | 2,150.0 ms | — | — | 1,920.9 ms | **4,071.0 ms** |
| **SMTP Email Action** (TLS Dispatch via Gmail) | 0.13 ms | 2,796.0 ms | — | — | 3,966.8 ms | **6,762.9 ms** |
| **RAG Cold Start** (Initial SentenceTransformer Load) | 0.10 ms | 2,572.8 ms | 30,505.9 ms* | — | — | **33,091.4 ms\*** |

*\*Note: Cold start latency occurs only once per process lifecycle when the local embedding model is initialized lazily. Pre-warming eliminates this delay.*

---

## 3. Component Deep Dive & Bottleneck Analysis

### 3.1 Prompt Construction (`prompt_construction_ms`)
* **Typical Latency**: **0.08 ms – 0.35 ms** (< 1ms)
* **Performance Assessment**: **Negligible Overhead**. The modular string synthesis in `src/core/prompts.py` executes in sub-millisecond time. In-memory dictionary lookups and string formatting introduce virtually zero bottleneck.

### 3.2 LLM Decision Routing (`llm_decision_ms`)
* **Typical Latency**: **1,850 ms – 2,600 ms** (Gemini 3.5 Flash-Lite over HTTPS)
* **Performance Assessment**:
  - `gemini-3.5-flash-lite` delivers stable responses within ~2 seconds for full structured JSON decision parsing (including domain classification, intent detection, stage tracking, slot extraction, and user response).
  - Standard `gemini-3.5-flash` demonstrated slightly higher latency (~2,800ms–3,500ms) and occasional HTTP 503 capacity limits during peak loads.
  - The fallback mechanism implemented in `src/core/llm.py` ensures seamless failover to `gemini-3.5-flash-lite` without call interruption.

### 3.3 Grounded Knowledge Retrieval (`rag_retrieval_ms`)
* **Warm In-Memory Vector Search**: **15 ms – 25 ms**
* **Live Redis Stack Hybrid Retrieval**: **8 ms – 18 ms** (dense vector cosine similarity + BM25 keyword score fusion)
* **Cold Start Penalty**: **~30,000 ms**
  - Occurred on the first turn requiring embeddings when `sentence-transformers/all-MiniLM-L6-v2` loaded torch tensors into memory on CPU.
  - Subsequent queries took **~20 ms** consistently.

### 3.4 MCP Tool & SMTP Execution (`tool_execution_ms`)
* **In-Memory Mock Tools** (`find_available_slots`, `create_calendar_event`): **~10 ms – 30 ms**
* **Live SMTP Email Provider** (`smtp.gmail.com:587`): **3,800 ms – 4,100 ms**
  - Root cause of SMTP latency: Synchronous DNS resolution, TLS socket handshake, authentication, and MIME message delivery over public internet.
  - In voice applications, blocking the spoken response until SMTP delivery completes introduces unnecessary silence.

---

## 4. Structured `[TURN DEBUG]` Log Format

The engine records standardized structured turn diagnostics at the completion of every turn for telemetry and real-time observability:

```text
================================================================================
[TURN DEBUG] Call ID: session-7fa3b190 | Direction: OUTBOUND | Domain: VAYVORA
--------------------------------------------------------------------------------
1. Caller Input: "yes"
2. Agent Response: "Wonderful! We specialize in custom enterprise AI calling agents that integrate seamlessly with your CRM and workflows. What specific operational challenges are you hoping to solve with AI voice automation?"
3. Intent & Stage:
   - Intent: ai_solution (Sub-intent: None)
   - Stage: purpose_discovery
   - Slots Extracted: {'company': 'Acme Corp'}
4. Grounded RAG Status:
   - Knowledge Required: False
   - Query: None
   - Citations: 0 chunks
5. Tool / MCP Actions:
   - Action Proposed: False
   - Tool Name: None
   - Execution Status: None
6. Latency Breakdown (Total: 2016.8ms):
   - prompt_construction: 0.12ms
   - llm_decision: 2016.68ms
   - rag_retrieval: 0.00ms
   - grounded_llm: 0.00ms
   - tool_execution: 0.00ms
================================================================================
```

---

## 5. Performance Optimization Roadmap for Production Telephony

To achieve sub-second conversational latency (< 800ms) for real-time telephony pipelines (Twilio, LiveKit, SIP), the following optimizations are recommended:

### 1. Server Startup Model Pre-Warming
* **Problem**: The local embedding model (`all-MiniLM-L6-v2`) lazy-loads on the first user query, introducing a 30-second stall.
* **Solution**: Execute a single dummy vector embedding during server bootstrap:
  ```python
  embedding_provider.embed_query("warmup_probe")
  ```
* **Impact**: Completely eliminates cold-start latency during live calls.

### 2. Asynchronous Background Task Dispatch for Heavy Tools (SMTP)
* **Problem**: SMTP email sending over TLS takes 3.8 to 4.1 seconds, blocking conversational progression.
* **Solution**: Decouple email dispatch using `asyncio.create_task()` or Celery/Redis queue. The agent immediately speaks:
  > *"I've queued that email to your address right now. While that sends, would you like to discuss...?"*
* **Impact**: Reduces tool turn latency from **~7,000ms** to **~2,200ms** (68% reduction).

### 3. Two-Tier LLM Architecture & Streaming Token Delivery
* **Problem**: Generating complete JSON decisions and full text responses in a single blocking call requires 1.8s–2.2s before speech synthesis can begin.
* **Solution**:
  - Stream tokens from Gemini (`streamGenerateContent`).
  - Feed the first sentence into TTS (Kokoro/ElevenLabs) as soon as punctuation (`.`, `?`, `!`) is detected (~300ms–500ms Time-To-First-Audio).
  - Use a small speculative router (or fine-tuned edge model) for instant filler responses (*"Certainly, let me check that for you..."*) while RAG executes concurrently.
* **Impact**: Perceived conversational latency drops below **600ms**.

### 4. Redis Vector Search Cache
* **Solution**: Cache frequently queried RAG results (e.g. course fees, office locations, services overview) in Redis with a 1-hour TTL.
* **Impact**: Cuts warm retrieval from 25ms to **< 2ms**.
