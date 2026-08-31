from typing import Optional

from src.voice.stt.whisper.adapter import STTAdapter
from src.voice.stt.common.schemas import TranscriptResult


class STTSession:
    """
    Represents the STT state for ONE voice call.

    Every phone call should have its own STTSession.

    Call A
        â†“
    STTSession A

    Call B
        â†“
    STTSession B
    """

    def __init__(
        self,
        session_id: str,
        sample_rate: int = 16000,
        language: str | None = "en",
    ):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.language = language

        self.stt = STTAdapter(
            sample_rate=sample_rate,
            language=language,
        )

        self.active = True

    def process_audio(
        self,
        audio_chunk: bytes,
    ) -> list[TranscriptResult]:
        """
        Process audio belonging to this call.
        """

        if not self.active:
            raise RuntimeError(
                f"STT session {self.session_id} "
                "is no longer active."
            )

        return self.stt.process_audio(
            audio_chunk
        )

    def set_language(
        self,
        language: str | None,
    ):
        """
        Change language for this call.
        """

        self.language = language

        self.stt.set_language(
            language
        )

    def reset(self):
        """
        Clear current audio state.
        """

        self.stt.reset()

    def close(self):
        """
        Close the STT session.
        """

        self.reset()

        self.active = False
