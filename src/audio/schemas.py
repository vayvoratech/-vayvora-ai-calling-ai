"""Validated audio data models and result contracts for VAD, STT, and TTS engines.

Ensures audio metadata, timestamps, sample rates, confidence scores, and raw byte
buffers are typed and validated without leaking model-specific dependencies into
the conversation engine.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class AudioChunk(BaseModel):
    """Discrete frame or slice of raw audio data."""

    model_config = ConfigDict(extra="forbid")

    data: bytes = Field(..., description="Raw audio byte buffer")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, ge=1, le=2, description="Channel count (1=mono, 2=stereo)")
    format: str = Field(default="pcm16", description="Audio format/encoding (pcm16, wav, float32)")
    timestamp: float = Field(default=0.0, ge=0.0, description="Timeline offset in seconds")
    duration: float = Field(default=0.0, ge=0.0, description="Duration in seconds")
    is_speech: Optional[bool] = Field(default=None, description="VAD classification if evaluated")

    def model_post_init(self, __context: Any) -> None:
        """Automatically derive chunk duration if not explicitly provided for PCM16."""
        if self.duration <= 0.0 and self.format == "pcm16" and self.sample_rate > 0:
            bytes_per_sample = 2 * self.channels
            object.__setattr__(self, "duration", len(self.data) / (self.sample_rate * bytes_per_sample))


class SpeechSegment(BaseModel):
    """Contiguous block of speech audio accumulated between speech_start and speech_end."""

    model_config = ConfigDict(extra="forbid")

    audio_data: bytes = Field(..., description="Concatenated speech audio bytes")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Sample rate in Hz")
    channels: int = Field(default=1, ge=1, le=2, description="Channel count")
    start_time: float = Field(default=0.0, ge=0.0, description="Speech onset timestamp in seconds")
    end_time: float = Field(default=0.0, ge=0.0, description="Speech offset timestamp in seconds")
    duration: float = Field(default=0.0, ge=0.0, description="Total speech duration in seconds")
    format: str = Field(default="pcm16", description="Audio encoding format")
    chunks_count: int = Field(default=1, ge=1, description="Number of merged chunks")

    def model_post_init(self, __context: Any) -> None:
        if self.duration <= 0.0 and self.end_time >= self.start_time:
            object.__setattr__(self, "duration", round(self.end_time - self.start_time, 4))
        if self.duration <= 0.0 and self.format == "pcm16" and self.sample_rate > 0:
            bytes_per_sample = 2 * self.channels
            object.__setattr__(self, "duration", round(len(self.audio_data) / (self.sample_rate * bytes_per_sample), 4))


class VADResult(BaseModel):
    """Voice Activity Detection state result for an audio chunk."""

    model_config = ConfigDict(extra="forbid")

    is_speech: bool = Field(..., description="Whether speech was detected in this chunk")
    speech_start: bool = Field(default=False, description="True on the leading edge transition from silence to speech")
    speech_end: bool = Field(default=False, description="True on the trailing edge transition from speech to silence")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Raw speech probability from model")
    timestamp: float = Field(default=0.0, ge=0.0, description="Current audio timeline offset in seconds")
    duration: Optional[float] = Field(default=None, ge=0.0, description="Duration of the evaluated chunk")


class STTResult(BaseModel):
    """Validated transcription outcome from Speech-to-Text inference."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Transcribed textual utterance")
    language: Optional[str] = Field(default="en", description="Detected or configured language code")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score if reported by model"
    )
    start_time: float = Field(default=0.0, ge=0.0, description="Segment start timestamp in seconds")
    end_time: float = Field(default=0.0, ge=0.0, description="Segment end timestamp in seconds")
    is_final: bool = Field(default=True, description="True if transcription is final")
    provider: str = Field(default="faster-whisper", description="STT engine identifier")
    model: str = Field(default="base.en", description="Specific model variant")
    audio_duration: Optional[float] = Field(default=None, ge=0.0, description="Input audio duration in seconds")
    processing_time: Optional[float] = Field(default=None, ge=0.0, description="Inference latency in seconds")
    real_time_factor: Optional[float] = Field(
        default=None, ge=0.0, description="Processing time divided by audio duration"
    )

    def __str__(self) -> str:
        """Allow STTResult to be treated directly as a string for backward compatibility."""
        return self.text


class TTSResult(BaseModel):
    """Synthesized speech audio payload and generation metadata from Text-to-Speech."""

    model_config = ConfigDict(extra="forbid")

    audio_data: bytes = Field(..., description="Synthesized audio byte buffer (WAV/PCM)")
    sample_rate: int = Field(default=24000, ge=8000, le=48000, description="Synthesized audio sample rate in Hz")
    channels: int = Field(default=1, ge=1, le=2, description="Channel count")
    format: str = Field(default="wav", description="Audio container or encoding format")
    text: str = Field(default="", description="Original source text synthesized")
    duration: Optional[float] = Field(default=None, ge=0.0, description="Synthesized audio duration in seconds")
    processing_time: Optional[float] = Field(default=None, ge=0.0, description="Synthesis latency in seconds")
    real_time_factor: Optional[float] = Field(
        default=None, ge=0.0, description="Processing time divided by audio duration"
    )
    provider: str = Field(default="kokoro", description="TTS engine identifier")
    voice: str = Field(default="af_heart", description="Voice profile used")
