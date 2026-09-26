"""Central application configuration using pydantic-settings.

Defines validated configuration models supporting environment variables
and .env files for future components without connecting to external services.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the Unified AI Voice Agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Environment
    app_env: str = Field(
        default="development",
        alias="APP_ENV",
        description="Deployment environment (development, staging, production)",
    )
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Standard logging severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Gemini LLM Settings (Phase 3)
    gemini_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="GEMINI_API_KEY",
        description="Google Gemini API authentication token",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash",
        alias="GEMINI_MODEL",
        description="Gemini LLM model identifier",
    )
    gemini_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        alias="GEMINI_TIMEOUT_SECONDS",
        description="Timeout in seconds for Gemini API calls",
    )
    gemini_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        alias="GEMINI_TEMPERATURE",
        description="Sampling temperature for LLM generation",
    )
    # LLM Provider Settings
    llm_provider: str = Field(
        default="gemini",
        alias="LLM_PROVIDER",
        description="LLM provider identifier (gemini, huggingface)",
    )

    # Hugging Face / Qwen Settings
    hf_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="HF_API_KEY",
        description="Hugging Face API authentication token",
    )
    hf_model: str = Field(
        default="Qwen/Qwen3-32B",
        alias="HF_MODEL",
        description="Hugging Face model identifier",
    )
    hf_base_url: str = Field(
        default="https://router.huggingface.co/v1",
        alias="HF_BASE_URL",
        description="Hugging Face OpenAI-compatible inference endpoint",
    )
    hf_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        alias="HF_TIMEOUT_SECONDS",
        description="Timeout in seconds for Hugging Face API calls",
    )
    hf_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        alias="HF_TEMPERATURE",
        description="Sampling temperature for Qwen generation",
    )

    # Redis Stack Settings for Grounded RAG (Phase 4)
    redis_host: str = Field(
        default="localhost",
        alias="REDIS_HOST",
        description="Redis server hostname or IP address",
    )
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        alias="REDIS_PORT",
        description="Redis server TCP port",
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        alias="REDIS_DB",
        description="Redis logical database index",
    )
    redis_password: Optional[SecretStr] = Field(
        default=None,
        alias="REDIS_PASSWORD",
        description="Redis authentication password if configured",
    )
    redis_edusaas_index: str = Field(
        default="idx:edusaas_vdb",
        alias="REDIS_EDUSAAS_INDEX",
        description="Redis search index for EduSaaS documents",
    )
    redis_vayvora_index: str = Field(
        default="idx:vayvora_vdb",
        alias="REDIS_VAYVORA_INDEX",
        description="Redis search index for Vayvora documents",
    )
    rag_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="RAG_EMBEDDING_MODEL",
        description="Configured embedding model identifier",
    )
    rag_embedding_dimension: int = Field(
        default=384,
        ge=1,
        alias="RAG_EMBEDDING_DIMENSION",
        description="Dimension size of dense vector embeddings",
    )
    rag_chunk_size: int = Field(
        default=500,
        ge=50,
        alias="RAG_CHUNK_SIZE",
        description="Target maximum character length for chunking",
    )
    rag_chunk_overlap: int = Field(
        default=50,
        ge=0,
        alias="RAG_CHUNK_OVERLAP",
        description="Character overlap between consecutive chunks",
    )
    rag_top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        alias="RAG_TOP_K",
        description="Default number of chunks to retrieve",
    )
    rag_relevance_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        alias="RAG_RELEVANCE_THRESHOLD",
        description="Cosine similarity cutoff threshold for grounding",
    )

    # Model Context Protocol (MCP) Settings (Phase 5)
    mcp_server_url: Optional[str] = Field(
        default="http://localhost:8000/mcp",
        alias="MCP_SERVER_URL",
        description="Endpoint URL for external MCP tool server",
    )
    mcp_enabled: bool = Field(
        default=True,
        alias="MCP_ENABLED",
        description="Toggle MCP tool server connectivity",
    )

    # SMTP Email Settings
    smtp_host: str = Field(
        default="localhost",
        alias="SMTP_HOST",
        description="SMTP server hostname or IP address",
    )
    smtp_port: int = Field(
        default=587,
        ge=1,
        le=65535,
        alias="SMTP_PORT",
        description="SMTP server port",
    )
    smtp_username: Optional[str] = Field(
        default=None,
        alias="SMTP_USERNAME",
        description="SMTP authentication username",
    )
    smtp_password: Optional[SecretStr] = Field(
        default=None,
        alias="SMTP_PASSWORD",
        description="SMTP authentication password",
    )
    smtp_from_email: str = Field(
        default="noreply@vayvora.com",
        alias="SMTP_FROM_EMAIL",
        description="Default sender email address",
    )
    smtp_use_tls: bool = Field(
        default=True,
        alias="SMTP_USE_TLS",
        description="Whether to use STARTTLS for SMTP connection",
    )
    smtp_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        alias="SMTP_TIMEOUT_SECONDS",
        description="Timeout in seconds for SMTP operations",
    )

    # Local Audio Engine Settings: VAD, STT & TTS (Phase 7)
    # VAD Configuration
    vad_model_path: str = Field(
        default="models/silero_vad.onnx",
        alias="VAD_MODEL_PATH",
        description="Path to the Silero VAD ONNX model",
    )
    vad_provider: str = Field(
        default="silero",
        alias="VAD_PROVIDER",
        description="VAD provider identifier (silero, mock)",
    )
    vad_sample_rate: int = Field(
        default=16000,
        alias="VAD_SAMPLE_RATE",
        description="Audio sampling rate expected by VAD in Hz",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="VAD_THRESHOLD",
        description="Speech detection probability threshold",
    )
    vad_min_speech_duration: float = Field(
        default=0.25,
        ge=0.0,
        alias="VAD_MIN_SPEECH_DURATION",
        description="Minimum duration in seconds to consider speech started",
    )
    vad_min_silence_duration: float = Field(
        default=0.5,
        ge=0.0,
        alias="VAD_MIN_SILENCE_DURATION",
        description="Silence duration in seconds required to trigger speech end",
    )

    # STT Configuration
    stt_provider: str = Field(
        default="faster-whisper",
        alias="STT_PROVIDER",
        description="STT provider identifier (faster-whisper, mock)",
    )
    stt_model: str = Field(
        default="base.en",
        alias="STT_MODEL",
        description="Whisper model name or path",
    )
    stt_language: str = Field(
        default="en",
        alias="STT_LANGUAGE",
        description="Expected speech audio language code",
    )
    stt_device: str = Field(
        default="cpu",
        alias="STT_DEVICE",
        description="Compute device for STT inference (cpu, cuda)",
    )
    stt_compute_type: str = Field(
        default="int8",
        alias="STT_COMPUTE_TYPE",
        description="Quantization / compute type (int8, float16, float32)",
    )
    stt_beam_size: int = Field(
        default=5,
        ge=1,
        alias="STT_BEAM_SIZE",
        description="Beam search size for decoding",
    )

    # TTS Configuration
    tts_provider: str = Field(
        default="kokoro",
        alias="TTS_PROVIDER",
        description="TTS provider identifier (kokoro, piper, mock)",
    )
    tts_voice: str = Field(
        default="af_heart",
        alias="TTS_VOICE",
        description="Voice identifier for synthesis",
    )
    tts_language: str = Field(
        default="en-us",
        alias="TTS_LANGUAGE",
        description="Target synthesis language code",
    )
    tts_sample_rate: int = Field(
        default=24000,
        alias="TTS_SAMPLE_RATE",
        description="Synthesized audio sample rate in Hz",
    )
    tts_output_format: str = Field(
        default="wav",
        alias="TTS_OUTPUT_FORMAT",
        description="Synthesized audio container/codec format (wav, pcm)",
    )

    # Telephony Integration Settings (Phase 9)
    telephony_transport: str = Field(
        default="mock",
        alias="TELEPHONY_TRANSPORT",
        description="Telephony media transport type (mock, websocket, sip)",
    )
    telephony_endpoint: Optional[str] = Field(
        default=None,
        alias="TELEPHONY_ENDPOINT",
        description="External telephony media stream or signaling endpoint URL",
    )
    telephony_connection_timeout: float = Field(
        default=10.0,
        ge=0.5,
        le=60.0,
        alias="TELEPHONY_CONNECTION_TIMEOUT",
        description="Connection timeout in seconds for telephony media stream",
    )
    telephony_audio_sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        alias="TELEPHONY_AUDIO_SAMPLE_RATE",
        description="Target sample rate in Hz for incoming/outgoing telephony audio",
    )
    telephony_audio_channels: int = Field(
        default=1,
        ge=1,
        le=2,
        alias="TELEPHONY_AUDIO_CHANNELS",
        description="Number of audio channels for telephony stream (1=mono)",
    )
    telephony_frame_duration: float = Field(
        default=0.02,
        ge=0.005,
        le=0.1,
        alias="TELEPHONY_FRAME_DURATION",
        description="Duration in seconds of discrete media packet frames (typically 20ms)",
    )


    @property
    def redis_url(self) -> str:
        """Construct full Redis connection URI."""
        password = (
            f":{self.redis_password.get_secret_value()}@"
            if self.redis_password
            else ""
        )
        return f"redis://{password}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of loaded application settings."""
    return Settings()
