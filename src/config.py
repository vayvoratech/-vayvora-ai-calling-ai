"""Central application configuration using pydantic-settings.

Defines validated configuration models supporting environment variables
and .env files for future components without connecting to external services.
"""

from functools import lru_cache
from typing import Any, Dict, Optional
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the Unified AI Voice Agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_environment_variables(cls, data: Any) -> Any:
        """Map environment variables from old MCP project into current settings."""
        if isinstance(data, dict):
            if "MAIL_HOST_SMTP" in data and "SMTP_HOST" not in data and "smtp_host" not in data:
                data["smtp_host"] = data["MAIL_HOST_SMTP"]
            if "MAIL_PORT_SMTP" in data and "SMTP_PORT" not in data and "smtp_port" not in data:
                data["smtp_port"] = data["MAIL_PORT_SMTP"]
            if "MAIL_HOST_IMAP" in data and "IMAP_HOST" not in data and "imap_host" not in data:
                data["imap_host"] = data["MAIL_HOST_IMAP"]
            if "MAIL_PORT_IMAP" in data and "IMAP_PORT" not in data and "imap_port" not in data:
                data["imap_port"] = data["MAIL_PORT_IMAP"]
            if "MAIL_USER" in data:
                if "SMTP_USERNAME" not in data and "smtp_username" not in data:
                    data["smtp_username"] = data["MAIL_USER"]
                if "IMAP_USERNAME" not in data and "imap_username" not in data:
                    data["imap_username"] = data["MAIL_USER"]
            if "MAIL_PASS" in data:
                if "SMTP_PASSWORD" not in data and "smtp_password" not in data:
                    data["smtp_password"] = data["MAIL_PASS"]
                if "IMAP_PASSWORD" not in data and "imap_password" not in data:
                    data["imap_password"] = data["MAIL_PASS"]
            if "WHATSAPP_API_URL" in data and "OPENWA_BASE_URL" not in data and "openwa_base_url" not in data:
                data["openwa_base_url"] = data["WHATSAPP_API_URL"]
            if "WHATSAPP_API_TOKEN" in data and "OPENWA_API_KEY" not in data and "openwa_api_key" not in data:
                data["openwa_api_key"] = data["WHATSAPP_API_TOKEN"]
            if "EVENTS_FILE" in data and "CALENDAR_EVENTS_PATH" not in data and "calendar_events_path" not in data:
                data["calendar_events_path"] = data["EVENTS_FILE"]
        return data

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
    agent_name: Optional[str] = Field(
        default=None,
        alias="AGENT_NAME",
        description="Configured name/identity of the AI agent (e.g. 'Sarah')",
    )

    # Gemini LLM Settings (Phase 3)
    gemini_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="GEMINI_API_KEY",
        description="Google Gemini API authentication token",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash-lite",
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

    # IMAP Email Settings (for reading recent emails)
    imap_host: str = Field(
        default="imap.gmail.com",
        alias="IMAP_HOST",
        description="IMAP server hostname or IP address",
    )
    imap_port: int = Field(
        default=993,
        ge=1,
        le=65535,
        alias="IMAP_PORT",
        description="IMAP server port",
    )
    imap_username: Optional[str] = Field(
        default=None,
        alias="IMAP_USERNAME",
        description="IMAP authentication username",
    )
    imap_password: Optional[SecretStr] = Field(
        default=None,
        alias="IMAP_PASSWORD",
        description="IMAP authentication password",
    )
    imap_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        alias="IMAP_TIMEOUT_SECONDS",
        description="Timeout in seconds for IMAP operations",
    )

    # WhatsApp Messaging Settings (via OpenWA)
    openwa_base_url: str = Field(
        default="http://localhost:2785",
        alias="OPENWA_BASE_URL",
        description="Base URL for self-hosted OpenWA WhatsApp service",
    )
    openwa_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="OPENWA_API_KEY",
        description="Optional API key for OpenWA service",
    )
    whatsapp_session_id: str = Field(
        default="default",
        alias="WHATSAPP_SESSION_ID",
        description="Session ID for OpenWA WhatsApp instance",
    )

    # Calendar Storage Settings
    calendar_events_path: str = Field(
        default="data/events.json",
        alias="CALENDAR_EVENTS_PATH",
        description="Path to JSON file used for calendar event storage",
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

    # Noise Suppression & Real-Time Acoustic Hygiene Settings
    noise_suppression_enabled: bool = Field(
        default=True,
        alias="NOISE_SUPPRESSION_ENABLED",
        description="Enable WebRTC-based real-time noise suppression",
    )
    noise_suppression_level: int = Field(
        default=2,
        ge=0,
        le=3,
        alias="NOISE_SUPPRESSION_LEVEL",
        description="WebRTC noise suppression aggressiveness level (0=mild, 1=medium, 2=high, 3=very high)",
    )
    high_pass_filter_enabled: bool = Field(
        default=True,
        alias="HIGH_PASS_FILTER_ENABLED",
        description="Enable high-pass filter to strip sub-80Hz mechanical rumble and electrical hum",
    )
    noise_gate_enabled: bool = Field(
        default=True,
        alias="NOISE_GATE_ENABLED",
        description="Enable soft noise gate to suppress residual background acoustic floor during speech pauses",
    )
    noise_gate_threshold_db: float = Field(
        default=-42.0,
        alias="NOISE_GATE_THRESHOLD_DB",
        description="Threshold in dBFS below which soft noise gating is applied",
    )
    vad_start_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        alias="VAD_START_THRESHOLD",
        description="Silero VAD speech detection onset threshold",
    )
    vad_end_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        alias="VAD_END_THRESHOLD",
        description="Silero VAD speech termination threshold",
    )
    vad_min_speech_duration_ms: int = Field(
        default=120,
        ge=32,
        le=2000,
        alias="VAD_MIN_SPEECH_DURATION_MS",
        description="Minimum consecutive speech duration in milliseconds to trigger speech start",
    )
    vad_min_silence_duration_ms: int = Field(
        default=250,
        ge=64,
        le=2000,
        alias="VAD_MIN_SILENCE_DURATION_MS",
        description="Silence duration in milliseconds required to trigger speech end",
    )
    vad_pre_roll_ms: int = Field(
        default=160,
        ge=0,
        le=1000,
        alias="VAD_PRE_ROLL_MS",
        description="Pre-roll buffer duration in milliseconds to preserve initial speech consonants",
    )
    pre_stt_min_duration_ms: int = Field(
        default=150,
        ge=50,
        le=2000,
        alias="PRE_STT_MIN_DURATION_MS",
        description="Minimum speech duration in milliseconds required before sending audio to STT",
    )
    pre_stt_min_rms: float = Field(
        default=120.0,
        ge=0.0,
        alias="PRE_STT_MIN_RMS",
        description="Minimum RMS energy required before sending audio segment to STT",
    )

    # STT Configuration (Groq Whisper Cloud STT)
    groq_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="GROQ_API_KEY",
        description="Groq API key for cloud Whisper speech-to-text",
    )
    stt_provider: str = Field(
        default="groq",
        alias="STT_PROVIDER",
        description="STT provider identifier (groq, mock, faster-whisper)",
    )
    stt_model: str = Field(
        default="whisper-large-v3-turbo",
        alias="STT_MODEL",
        description="Whisper model name or path",
    )
    groq_stt_model: str = Field(
        default="whisper-large-v3-turbo",
        alias="GROQ_STT_MODEL",
        description="Groq STT model override",
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

    # TTS Configuration (Deepgram Streaming TTS)
    deepgram_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="DEEPGRAM_API_KEY",
        description="Deepgram API key for real-time streaming speech synthesis",
    )
    tts_provider: str = Field(
        default="deepgram",
        alias="TTS_PROVIDER",
        description="TTS provider identifier (deepgram, mock, kokoro, piper)",
    )
    tts_voice: str = Field(
        default="aura-asteria-en",
        alias="TTS_VOICE",
        description="Voice identifier for synthesis",
    )
    deepgram_tts_model: str = Field(
        default="aura-asteria-en",
        alias="DEEPGRAM_TTS_MODEL",
        description="Deepgram TTS model voice identifier",
    )
    deepgram_tts_encoding: str = Field(
        default="linear16",
        alias="DEEPGRAM_TTS_ENCODING",
        description="Deepgram TTS audio encoding",
    )
    deepgram_tts_sample_rate: int = Field(
        default=48000,
        alias="DEEPGRAM_TTS_SAMPLE_RATE",
        description="Deepgram TTS audio sample rate",
    )
    tts_language: str = Field(
        default="en-us",
        alias="TTS_LANGUAGE",
        description="Target synthesis language code",
    )
    tts_sample_rate: int = Field(
        default=48000,
        alias="TTS_SAMPLE_RATE",
        description="Synthesized audio sample rate in Hz",
    )
    tts_output_format: str = Field(
        default="linear16",
        alias="TTS_OUTPUT_FORMAT",
        description="Synthesized audio container/codec format (linear16, wav, pcm)",
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


    # PostgreSQL Database Configuration
    postgres_host: str = Field(
        default="localhost",
        alias="POSTGRES_HOST",
        description="Host for PostgreSQL database",
    )
    postgres_port: int = Field(
        default=5434,
        ge=1,
        le=65535,
        alias="POSTGRES_PORT",
        description="TCP port for PostgreSQL database",
    )
    postgres_db: str = Field(
        default="aicall",
        alias="POSTGRES_DB",
        description="Database name for PostgreSQL",
    )
    postgres_user: str = Field(
        default="postgres",
        alias="POSTGRES_USER",
        description="Username for PostgreSQL database",
    )
    postgres_password: Optional[SecretStr] = Field(
        default=None,
        alias="POSTGRES_PASSWORD",
        description="Password for PostgreSQL database",
    )
    database_url: Optional[str] = Field(
        default=None,
        alias="DATABASE_URL",
        description="Full connection URL for PostgreSQL database",
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

    @property
    def resolved_database_url(self) -> str:
        """Construct or return full PostgreSQL connection URI."""
        if self.database_url:
            return self.database_url
        pwd = (
            f":{self.postgres_password.get_secret_value()}"
            if self.postgres_password
            else ""
        )
        return f"postgresql+asyncpg://{self.postgres_user}{pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def asyncpg_dsn(self) -> str:
        """Construct raw asyncpg-compatible DSN."""
        url = self.resolved_database_url
        if url.startswith("postgresql+asyncpg://"):
            return "postgresql://" + url[len("postgresql+asyncpg://"):]
        return url


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of loaded application settings."""
    return Settings()
