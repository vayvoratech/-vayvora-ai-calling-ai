# Phase 8 — Pipecat Real-Time Voice Orchestration & Barge-In

This directory contains the real-time voice orchestration pipeline, turn lifecycle state machine, provider adapters, and barge-in interruption mechanisms for the unified AI voice agent:

```text
                    ┌──────────────┐
                    │ Audio Input  │ (Microphone / File / Mock)
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │     VAD      │ (Silero / Mock)
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │     STT      │ (faster-whisper / Mock)
                    └──────┬───────┘
                           ↓
                         TEXT
                           ↓
                  ┌──────────────────┐
                  │ ConversationEngine│ (Intent, RAG, MCP, Prompt Synthesis)
                  └────────┬─────────┘
                           ↓
                    ┌──────────────┐
                    │ Gemini / RAG │
                    │     / MCP    │
                    └──────┬───────┘
                           ↓
                         TEXT
                           ↓
                    ┌──────────────┐
                    │     TTS      │ (Kokoro / Piper / Mock)
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │ Audio Output │ (Speaker / Buffer / Sink)
                    └──────────────┘

             ←────── BARGE-IN ──────→
```

Pipecat owns **timing, frame routing, streaming, and interruption**.
`ConversationEngine` owns **conversation intelligence, state, RAG grounding, and MCP actions**.

---

## 1. Directory Structure

```text
src/voice/
├── __init__.py           # Package exports
├── pipeline.py           # Core real-time VoicePipeline frame ingestion loop
├── session.py            # VoiceSession lifecycle & deterministic mock transports
├── events.py             # Typed immutable voice lifecycle events
├── context.py            # TurnLifecycleState, CancellationToken, VoiceDiagnostics
├── adapters/             # Thin provider adapters
│   ├── __init__.py
│   ├── vad.py            # PipecatVADAdapter
│   ├── stt.py            # PipecatSTTAdapter
│   ├── tts.py            # PipecatTTSAdapter (cancellation & barge-in support)
│   └── llm.py            # PipecatConversationAdapter
└── README.md             # This reference guide
```

---

## 2. Turn Lifecycle State Machine

The voice orchestration pipeline operates through a deterministic turn state machine:

```text
LISTENING
    ↓ (VAD speech_start detected)
SPEECH_DETECTED
    ↓
CAPTURING (accumulating AudioChunks into SpeechSegment)
    ↓ (VAD speech_end detected after min_silence_duration)
TRANSCRIBING (STT inference)
    ↓ (non-empty transcript)
THINKING (ConversationEngine intent + RAG + MCP)
    ↓ (agent response synthesized)
SPEAKING (TTS synthesis & audio output)
    ↓ (TTS playback completed)
LISTENING
```

Additional states:
- **`INTERRUPTED`**: Caller begins speaking while agent is speaking or thinking (barge-in).
- **`COMPLETED`**: Call has ended gracefully (e.g. caller said goodbye).
- **`ERROR`**: Unhandled provider or transport exception.

---

## 3. Barge-In & Interruption Handling

When the caller starts speaking while the assistant is in `SPEAKING` or `THINKING` state:
1. **Immediate Detection**: VAD fires `speech_start=True`.
2. **Cancellation Token Invalidation**: `cancellation_token.cancel()` is called.
3. **Generation Advancement**: `current_generation_id` increments, invalidating all outstanding audio frames and LLM chunks from the current generation.
4. **TTS Interruption**: `tts_adapter.cancel_active_synthesis()` cancels ongoing neural speech synthesis.
5. **Output Sink Purge**: `audio_output.clear()` purges queued assistant audio bytes from the output buffer.
6. **Telemetry & Event**: Fires `CallerInterrupted` event and increments `interruption_count`.
7. **Transition to Capture**: Pipeline state immediately transitions to `CAPTURING` to capture caller's incoming speech.

---

## 4. Race Condition Protection

The architecture explicitly protects against asynchronous timing race conditions:

- **Case 1: Late TTS Frame Arrives After Interruption**:
  The TTS adapter checks `cancellation_token.is_valid_generation(generation_id)`. If the generation was cancelled, the frame is rejected with `StaleGenerationError` and dropped without reaching the audio sink.
- **Case 2: Gemini Response Still Streaming on Interruption**:
  The conversation adapter verifies generation tokens before and after engine invocation. If cancelled, results are discarded.
- **Case 3: Previous Turn Finishes Late While New Turn Has Started**:
  Each turn operates with a unique `generation_id` and `turn_id`. Out-of-order completions cannot overwrite new conversation turns.
- **Case 4: MCP Action Executing When Interrupted**:
  External business mutations (calendar event creation, lead status updates, emails) are **never blindly aborted** unless the tool contract explicitly supports safe cancellation. The action completes and verifies in the background, while the voice pipeline captures the new caller turn.

---

## 5. Developer Diagnostics & Telemetry

`VoiceDiagnostics` records actual measured latencies and metrics:
- **VAD**: `speech_start_time`, `speech_end_time`, `speech_duration`
- **STT**: `transcription_latency`, `audio_duration`, `stt_rtf`
- **LLM**: `first_token_latency`, `total_generation_latency`
- **TTS**: `first_audio_latency`, `total_synthesis_latency`, `tts_rtf`
- **Pipeline**: `turn_latency`, `interruption_count`, `cancelled_generations_count`, `dropped_stale_frames_count`, `errors`

---

## 6. Mock Voice Pipeline & Testing

Deterministic mock providers and transports allow 100% test coverage without hardware:
- `MockAudioInput`: Simulates microphone, file, or packet stream.
- `MockAudioOutput`: In-memory sink recording synthesized WAV bytes with `.clear()` for barge-in testing.
- `MockPipecatTransport`: Bidirectional simulated transport.

All unit tests run on CPU in seconds with zero network calls and zero model weight downloads.

---

## 7. Telephony Transport Integration (Phase 9)

Voice sessions and real-time pipelines connect directly to external telephony media transports (e.g. WebSocket audio streams, mock transports) via the **Phase 9 Telephony Layer** in `src/telephony/`:
- `TelephonyVoiceAdapter` mediates bidirectional PCM streaming and barge-in.
- `TelephonyAudioConverter` handles 8kHz <-> 16kHz linear resampling and frame validation.
- `CallSessionRepository` persists structured, sanitized `CallSummary` records upon call conclusion.

See [`src/telephony/README.md`](file:///C:/Users/Pawanganesh/Desktop/voice/src/telephony/README.md) for complete details.

---

## 8. Known Limitations & Production Telephony Boundary

> [!IMPORTANT]
> **Production Telephony Boundary**:
> This codebase implements the application-side voice orchestration and transport adapters. Production carrier infrastructure (SIP trunks, PBX, carrier signaling, phone-number provisioning, and Twilio account infrastructure) is managed externally.

