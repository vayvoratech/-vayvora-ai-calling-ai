"""Custom exceptions for LLM operations and decision validation."""


class LLMError(Exception):
    """Base exception for all LLM provider and generation errors."""

    pass


class LLMAuthenticationError(LLMError):
    """Raised when API credentials or keys are missing or rejected."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when an LLM network request times out."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns a rate limit (HTTP 429) status."""

    pass


class LLMMalformedResponseError(LLMError):
    """Raised when the LLM produces invalid, unparseable, or non-JSON output."""

    pass


class DecisionValidationError(LLMError):
    """Raised when a structured decision fails domain, intent, or slot validation."""

    pass


class ToolError(Exception):
    """Base exception for external tool and MCP execution errors."""

    pass


class UnsupportedActionError(ToolError):
    """Raised when an unapproved or unknown tool action is proposed."""

    pass


class InvalidToolArgumentsError(ToolError):
    """Raised when required arguments for an action are missing or invalid."""

    pass


class ToolExecutionError(ToolError):
    """Raised when external tool execution fails or returns an error."""

    pass


class ToolVerificationError(ToolError):
    """Raised when a tool execution fails post-execution verification."""

    pass


class MCPUnavailableError(ToolError):
    """Raised when the MCP server or transport is unreachable."""

    pass


class MCPTimeoutError(ToolError):
    """Raised when an MCP tool request times out."""

    pass


class MCPAuthenticationError(ToolError):
    """Raised when authentication to the MCP server fails."""

    pass


# ==============================================================================
# Audio, VAD, STT & TTS Exceptions (Phase 7)
# ==============================================================================

class AudioError(Exception):
    """Base exception for all local audio processing and pipeline errors."""

    pass


class AudioProviderInitializationError(AudioError):
    """Raised when an audio engine provider fails to initialize."""

    pass


class ModelUnavailableError(AudioError):
    """Raised when an audio model (Silero, Whisper, Kokoro, Piper) is not installed or available."""

    pass


class InvalidAudioDataError(AudioError):
    """Raised when audio payload is empty, corrupted, or incompatible."""

    pass


class VADError(AudioError):
    """Base exception for Voice Activity Detection errors."""

    pass


class VADProcessingError(VADError):
    """Raised when an audio frame cannot be evaluated by the VAD engine."""

    pass


class STTError(AudioError):
    """Base exception for Speech-to-Text errors."""

    pass


class STTTranscriptionError(STTError):
    """Raised when speech transcription fails."""

    pass


class TTSError(AudioError):
    """Base exception for Text-to-Speech synthesis errors."""

    pass


class TTSSynthesisError(TTSError):
    """Raised when text synthesis fails."""

    pass


class AudioPipelineError(AudioError):
    """Raised when audio framing or segmentation fails in the test pipeline."""

    pass


# ==============================================================================
# Phase 8: Voice Orchestration Errors
# ==============================================================================

class VoiceSessionError(Exception):
    """Base exception for all real-time voice session errors."""

    pass


class VoicePipelineError(VoiceSessionError):
    """Raised when frame routing, transport, or timing fails in the voice pipeline."""

    pass


class BargeInInterruptionError(VoiceSessionError):
    """Raised when active agent speech synthesis is cancelled due to caller interruption."""

    pass


class StaleGenerationError(VoiceSessionError):
    """Raised when an out-of-date generation chunk is discarded."""

    pass


# ==============================================================================
# Phase 9: Telephony Transport & Persistence Errors
# ==============================================================================

class TelephonyError(Exception):
    """Base exception for all telephony transport adapter errors."""

    pass


class TelephonyConnectionError(TelephonyError):
    """Raised when connecting or communicating with the telephony media stream fails."""

    pass


class TelephonyAudioIncompatibleError(TelephonyError):
    """Raised when telephony audio format, sample rate, or channels are incompatible."""

    pass


class TelephonyPacketError(TelephonyError):
    """Raised when telephony packet payload or framing is corrupted or malformed."""

    pass


class SessionPersistenceError(Exception):
    """Raised when call summary or session record cannot be persisted."""

    pass



