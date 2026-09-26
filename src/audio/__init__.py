"""Local Audio Engines: VAD, STT, and TTS provider implementations and data contracts.

Provides local, replaceable audio inference abstractions for Silero VAD,
faster-whisper STT, and Kokoro/Piper TTS, alongside deterministic mock providers
for unit testing and local development.
"""

from src.audio.pipeline import AudioPipeline
from src.audio.schemas import AudioChunk, SpeechSegment, STTResult, TTSResult, VADResult
from src.audio.stt import BaseSTTProvider, FasterWhisperSTTProvider, MockSTTProvider
from src.audio.tts import (
    BaseTTSProvider,
    KokoroTTSProvider,
    MockTTSProvider,
    PiperTTSProvider,
    create_pcm_wav_bytes,
)
from src.audio.vad import BaseVADProvider, MockVADProvider, SileroVADProvider

__all__ = [
    # Schemas
    "AudioChunk",
    "SpeechSegment",
    "VADResult",
    "STTResult",
    "TTSResult",
    # VAD
    "BaseVADProvider",
    "MockVADProvider",
    "SileroVADProvider",
    # STT
    "BaseSTTProvider",
    "MockSTTProvider",
    "FasterWhisperSTTProvider",
    # TTS
    "BaseTTSProvider",
    "MockTTSProvider",
    "KokoroTTSProvider",
    "PiperTTSProvider",
    "create_pcm_wav_bytes",
    # Pipeline
    "AudioPipeline",
]
