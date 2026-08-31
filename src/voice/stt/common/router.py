from typing import Optional

from src.voice.stt.common.provider import STTProvider
from src.voice.stt.common.schemas import Transcript


class STTRouter:

    def __init__(
        self,
        whisper_provider: STTProvider,
        parakeet_provider: Optional[STTProvider] = None,
    ):
        self.whisper = whisper_provider
        self.parakeet = parakeet_provider

    def get_provider(
        self,
        language: Optional[str],
    ) -> STTProvider:

        if language == "en" and self.parakeet is not None:
            return self.parakeet

        return self.whisper

    def process_audio(
        self,
        audio_chunk: bytes,
        language: Optional[str],
    ) -> list[Transcript]:

        provider = self.get_provider(language)

        return provider.process_audio(audio_chunk)

    def set_language(
        self,
        language: Optional[str],
    ) -> None:

        provider = self.get_provider(language)

        provider.set_language(language)

    def reset(
        self,
        language: Optional[str],
    ) -> None:

        provider = self.get_provider(language)

        provider.reset()