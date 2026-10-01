"""VAD service export alias."""

from src.voice.audio_services.vad_services import VADEvent, VADEventType, VADService

__all__ = ["VADService", "VADEvent", "VADEventType"]
