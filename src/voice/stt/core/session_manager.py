from src.voice.stt.core.session import STTSession


class STTSessionManager:
    """
    Manages STT sessions for multiple simultaneous calls.
    """

    def __init__(self):
        self.sessions: dict[str, STTSession] = {}

    def create_session(
        self,
        session_id: str,
        language: str | None = "en",
        sample_rate: int = 16000,
    ) -> STTSession:

        if session_id in self.sessions:
            raise ValueError(
                f"STT session already exists: "
                f"{session_id}"
            )

        session = STTSession(
            session_id=session_id,
            sample_rate=sample_rate,
            language=language,
        )

        self.sessions[session_id] = session

        return session

    def get_session(
        self,
        session_id: str,
    ) -> STTSession:

        if session_id not in self.sessions:
            raise KeyError(
                f"STT session not found: "
                f"{session_id}"
            )

        return self.sessions[session_id]

    def remove_session(
        self,
        session_id: str,
    ):

        session = self.sessions.pop(
            session_id,
            None,
        )

        if session:
            session.close()

    def active_sessions(self) -> int:
        return len(self.sessions)
