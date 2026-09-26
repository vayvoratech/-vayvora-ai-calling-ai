# Phase 7 — Local Audio Engines: VAD, STT & TTS

This directory contains local audio processing abstractions and provider implementations for the unified AI voice agent:
- **Voice Activity Detection (VAD)**: Silero VAD (ONNX Runtime)
- **Speech-to-Text (STT)**: faster-whisper (CTranslate2)
- **Text-to-Speech (TTS)**: Kokoro TTS / Piper TTS (ONNX)
- **Deterministic Mock Engines**: Zero-download test providers for fast local testing and CI/CD pipelines.

---

## 1. Architecture & Provider Pattern

The audio layer interfaces strictly through decoupled provider contracts (`VADProvider`, `STTProvider`, `TTSProvider` in `src.core.interfaces`).

```text
Microphone / Audio In
          ↓
     [AudioChunk]
          ↓
     VADProvider (Silero / Mock)
          ↓ (Accumulates speech frames; filters noise)
    [SpeechSegment]
          ↓
     STTProvider (faster-whisper / Mock)
          ↓
      STTResult (text utterance)
          ↓
   ConversationEngine (Receives text, returns text)
          ↓
      LLM Decision (user_facing_response)
          ↓
     TTSProvider (Kokoro / Piper / Mock)
          ↓
      TTSResult (RIFF WAV 16-bit PCM bytes)
          ↓
Audio Speaker / Telephony Sink
```

### Key Architectural Principles
1. **Engine Decoupling**: Model-specific code, ONNX tensors, sample rate buffers, and audio frames never leak into the core `ConversationEngine` or `ConversationState`.
2. **Deterministic Mock Providers**: `MockVADProvider`, `MockSTTProvider`, and `MockTTSProvider` enable complete unit test execution without GPU hardware or downloading gigabytes of model checkpoints.
3. **Interruption & Barge-in Readiness**: `TTSProvider.cancel_current_synthesis()` provides a cancellation token that stops active synthesis or streaming turns when the caller interrupts the agent.

---

## 2. Core Data Contracts (`src/audio/schemas.py`)

- **`AudioChunk`**: Discrete slice of raw PCM audio data (sample rate, channels, format, duration, timestamp).
- **`SpeechSegment`**: Contiguous block of merged speech audio assembled between `speech_start` and `speech_end`.
- **`VADResult`**: Voice activity detection status (`is_speech`, `speech_start`, `speech_end`, `confidence`, `timestamp`, `duration`).
- **`STTResult`**: Transcribed utterance (`text`, `language`, `confidence`, `audio_duration`, `processing_time`, `real_time_factor`, `is_final`).
- **`TTSResult`**: Synthesized speech payload (`audio_data` bytes, `sample_rate`, `format="wav"`, `text`, `duration`, `processing_time`, `real_time_factor`, `voice`).

---

## 3. Supported Engines

### Voice Activity Detection (VAD)
- **Silero VAD (`SileroVADProvider`)**:
  - Sample Rate: 16,000 Hz mono PCM16.
  - Runtime: ONNX Runtime (`onnxruntime`).
  - Hysteresis State Machine:
    - `min_speech_duration` (default 0.25s): Filters short noise spikes.
    - `min_silence_duration` (default 0.50s): Prevents premature turn termination during brief natural pauses.
- **Mock VAD (`MockVADProvider`)**:
  - Deterministic probability sequencing and RMS energy thresholding.

### Speech-to-Text (STT)
- **faster-whisper (`FasterWhisperSTTProvider`)**:
  - Fast, CTranslate2-optimized Whisper inference.
  - Configurable compute type (`int8`, `float16`, `float32`) and device (`cpu` or `cuda`).
  - Default Model: `base.en`.
- **Mock STT (`MockSTTProvider`)**:
  - Deterministic transcription queues, latency simulation, and Real-Time Factor (RTF) calculation.

### Text-to-Speech (TTS)
- **Kokoro TTS (`KokoroTTSProvider`)**:
  - Natural, expressive neural speech synthesis.
  - Default Voice: `af_heart` (American English).
  - Sample Rate: 24,000 Hz.
- **Piper TTS (`PiperTTSProvider`)**:
  - Ultra-fast local ONNX voice synthesis fallback (sample rate: 22,050 Hz).
- **Mock TTS (`MockTTSProvider`)**:
  - Synthesizes 100% valid, playable 16-bit PCM RIFF WAV containers using standard library `wave` and numpy sine frequencies.

---

## 4. Configuration Settings

Audio configuration is managed via `.env` or `src/config.py`:

```bash
# VAD Configuration
VAD_PROVIDER=mock          # "silero" or "mock"
VAD_SAMPLE_RATE=16000
VAD_THRESHOLD=0.5
VAD_MIN_SPEECH_DURATION=0.25
VAD_MIN_SILENCE_DURATION=0.5

# STT Configuration
STT_PROVIDER=mock          # "faster-whisper" or "mock"
STT_MODEL=base.en
STT_LANGUAGE=en
STT_DEVICE=cpu             # "cpu" or "cuda"
STT_COMPUTE_TYPE=int8
STT_BEAM_SIZE=5

# TTS Configuration
TTS_PROVIDER=mock          # "kokoro", "piper", or "mock"
TTS_VOICE=af_heart
TTS_LANGUAGE=en-us
TTS_SAMPLE_RATE=24000
TTS_OUTPUT_FORMAT=wav
```

---

## 5. Streamlit Testing Workbench

In the Streamlit workbench (`src/ui/app.py`):
1. **Sidebar Status**: Displays active modes for all subsystems:
   `LLM: MOCK | RAG: MOCK | Tools: MOCK`
   `VAD: MOCK | STT: MOCK | TTS: MOCK`
2. **Audio Engines Test Bench**: Switching conversation mode or opening the Audio Bench tab provides interactive controls:
   - **TTS**: Enter any sentence, synthesize audio, inspect latency/RTF, and play the WAV audio directly in the browser via `st.audio()`.
   - **STT**: Transcribe audio frames and verify speech recognition output.
   - **VAD**: Evaluate speech vs. silence chunks and verify edge detection triggers.
3. **Debug Inspection Panel**: The "🎙️ Audio Engines" tab inspects the live runtime configuration, active provider classes, and sample rates.

---

## 6. Windows Installation Instructions

To install dependencies for live local models on Windows:

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install audio dependencies
pip install soundfile onnxruntime

# 3. Optional: Install faster-whisper for live STT
pip install faster-whisper

# 4. Optional: Install kokoro for live TTS
pip install kokoro
```
