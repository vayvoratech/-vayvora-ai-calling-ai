# Phase 9 — External Telephony Transport Integration & End-to-End Validation

This package provides the application-side telephony transport adapters, format converters, and session persistence repository for the unified AI voice agent:

```text
External Telephony Gateway / Media Stream (WebSocket / RTP)
                       ↕ (8kHz or 16kHz PCM Frames)
         ┌─────────────────────────────┐
         │     TelephonyTransport      │ (GenericWebSocket / Mock)
         └─────────────┬───────────────┘
                       ↕ (telephony packets)
         ┌─────────────┴───────────────┐
         │   TelephonyAudioConverter   │ (Validation, 8kHz ↔ 16kHz Linear Resampling)
         └─────────────┬───────────────┘
                       ↕ (AudioChunk)
         ┌─────────────┴───────────────┐
         │    TelephonyVoiceAdapter    │ (Inbound/Outbound, Disconnect, Barge-In)
         └──────┬───────────────┬──────┘
                │               │
                ↓               ↓
         ┌──────────────┐┌──────────────────────┐
         │ VoiceSession ││ CallSessionRepository│
         │ (Pipecat)    ││ (Sanitized Summaries)│
         └──────────────┘└──────────────────────┘
```

The application owns only the **transport adapter and media bridge**.
Carrier switching, SIP trunking, PBX, and phone provisioning are managed externally.

---

## 1. Directory Structure

```text
src/telephony/
├── __init__.py           # Package exports
├── models.py             # TelephonyEvent, CallSummary, CallObservabilityDiagnostics
├── interfaces.py         # TelephonyTransport & CallSessionRepository contracts
├── audio.py              # TelephonyAudioConverter (validation, 8kHz ↔ 16kHz resampling)
├── transport.py          # MockTelephonyTransport & GenericWebSocketTelephonyTransport
├── repository.py         # MockCallSessionRepository (in-memory persistence)
├── adapter.py            # TelephonyVoiceAdapter & TelephonyAudioOutputSink
└── README.md             # This reference guide
```

---

## 2. Core Components

### `TelephonyTransport` (Interface & Implementations)
- **`MockTelephonyTransport`**: Deterministic mock transport for isolated unit and E2E testing without network dependencies. Provides `feed_audio()`, `receive_audio()`, and `sent_audio_frames` verification.
- **`GenericWebSocketTelephonyTransport`**: Production-ready WebSocket media streaming transport connecting to external media gateways.

### `TelephonyAudioConverter`
- **Validation**: Enforces 16-bit PCM encoding, valid channel counts (mono/stereo), non-empty buffers, and frame boundary alignments (no odd byte counts).
- **Linear Resampling**: Resamples telephony audio (e.g. standard 8kHz G.711/PCM) to the pipeline sample rate (16kHz), and downsamples synthesized outbound 16kHz TTS chunks to 8kHz telephony frames.
- **Header Stripping**: Automatically strips 44-byte RIFF WAV headers when delivering outbound frames to ensure clean raw PCM byte streams over telephony transports.

### `TelephonyVoiceAdapter`
- **Inbound Calls**: Handles unknown callers with brand-safe defaults, creates inbound `ConversationState`, and connects the real-time `VoicePipeline`.
- **Outbound Calls**: Enriches state with campaign context (`campaign_id`, `campaign_objective`) and caller metadata while preserving responsiveness if caller preempts campaign goals.
- **Media Streaming**: Ingests packets via `process_incoming_packet()`, passing them to VAD and dispatching synthesized TTS audio back through `TelephonyTransport`.
- **Barge-In Coordination**: Immediately flushes the output sink and cancels pending synthesis upon caller speech detection.
- **Disconnect & Teardown**: Gracefully stops the voice session and persists an end-of-call summary to the repository.

### `CallSessionRepository` & `CallSummary`
- Persists sanitized records upon call completion with:
  - Call ID, Session ID, direction, and business domain
  - Caller profile (without raw audio or secrets)
  - Intent progression history
  - CRM qualification status
  - Verified completed actions and any failed actions
  - Accurate call duration and termination reason

---

## 3. Disconnect Semantics

Calls conclude under three distinct conditions:
1. **Remote Caller Hangup**: Remote carrier disconnects transport. Recorded as `remote_caller_disconnected`.
2. **Network Dropped**: Carrier socket closes abruptly. Recorded as `network_disconnect`.
3. **Natural Farewell**: Caller says explicit goodbye ("goodbye", "have a great day"). Conversation engine sets `termination_requested = True` and records `completed_goodbye`.

---

## 4. Testing & Verification

The telephony module and full end-to-end stack are verified with automated test suites:
- `tests/test_telephony.py`: 13 unit tests covering transport buffering, audio converter resampling, packet validation, and repository operations.
- `tests/test_e2e.py`: 42 comprehensive end-to-end scenarios covering inbound, outbound, conversation semantics, grounded RAG, MCP verified actions, barge-in, disconnect persistence, and failure recovery.
- `tests/test_telephony_live.py`: Optional live WebSocket tests that automatically skip when `TELEPHONY_ENDPOINT` is not configured.

Run all tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_telephony.py tests/test_e2e.py -v
```

---

## 5. Critical Infrastructure Boundary

> [!IMPORTANT]
> **What This Codebase Owns**:
> - Telephony media packet ingestion and conversion.
> - Voice pipeline orchestration, barge-in interruption, and turn transitions.
> - Conversation state, intent routing, grounded RAG, and MCP verified actions.
> - Call summary assembly and persistence boundary.
>
> **What Remains External**:
> - SIP signaling servers, PBX, Asterisk, and FreeSWITCH switches.
> - Carrier trunking, PSTN phone-number routing, and WebRTC signaling gateways.
> - Twilio / external CPaaS account administration and billing infrastructure.
