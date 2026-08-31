from abc import ABC, abstractmethod
from typing import Optional

from src.voice.stt.common.schemas import Transcript


class STTProvider(ABC):

    @abstractmethod
    def process_audio(
        self,
        audio_chunk: bytes,
    ) -> list[Transcript]:
        raise NotImplementedError

    @abstractmethod
    def set_language(
        self,
        language: Optional[str],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError