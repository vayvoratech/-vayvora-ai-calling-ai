from dataclasses import dataclass


@dataclass
class STTConfig:
    """
    Central configuration for the STT system.
    """

    sample_rate: int = 16000

    # VAD
    vad_threshold: float = 0.25
    silence_duration: float = 0.8

    # Whisper
    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "int8"

    # Language
    language: str | None = "en"

    # Streaming
    partial_enabled: bool = True
    partial_interval: float = 0.8
    minimum_partial_duration: float = 0.8