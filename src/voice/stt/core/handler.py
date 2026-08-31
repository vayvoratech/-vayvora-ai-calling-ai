from src.voice.stt.core.session_manager import STTSessionManager


class STTHandler:
    """
    WebSocket-facing STT handler.

    The Voice Session Service should use this class
    instead of accessing the STT internals directly.
    """

    def __init__(self):
        self.session_manager = STTSessionManager()

    def start_session(
        self,
        session_id: str,
        language: str | None = "en",
        sample_rate: int = 16000,
    ):
        """
        Create an STT session for a call.
        """

        return self.session_manager.create_session(
            session_id=session_id,
            language=language,
            sample_rate=sample_rate,
        )

    def process_audio(
        self,
        session_id: str,
        audio_chunk: bytes,
    ):
        """
        Process one WebSocket audio message.
        """

        session = self.session_manager.get_session(
            session_id
        )

        results = session.process_audio(
            audio_chunk
        )

        return [
            result.to_dict()
            for result in results
        ]

    def set_language(
        self,
        session_id: str,
        language: str | None,
    ):
        """
        Change the language for an active call.
        """

        session = self.session_manager.get_session(
            session_id
        )

        session.set_language(language)

    def end_session(
        self,
        session_id: str,
    ):
        """
        End a call's STT session.
        """

        self.session_manager.remove_session(
            session_id
        )

    def active_sessions(self) -> int:
        return self.session_manager.active_sessions()
