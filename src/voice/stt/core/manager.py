import asyncio
from typing import Awaitable, Callable

from src.voice.stt.core.session import STTSession
from src.voice.stt.common.schemas import TranscriptResult
from src.voice.stt.core.worker import STTWorker


TranscriptCallback = Callable[
    [str, TranscriptResult],
    Awaitable[None],
]


class STTManager:
    """
    High-level STT manager.

    Responsibilities:
        - Manage call sessions
        - Receive audio
        - Queue audio
        - Run STT worker
        - Deliver transcripts
        - Clean up sessions

    Architecture:

        WebSocket
            â†“
        STTManager
            â†“
        STTWorker
            â†“
        STTSession
            â†“
        STT Pipeline
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        language: str | None = "en",
        max_queue_size: int = 100,
    ):
        self.sample_rate = sample_rate
        self.language = language

        self.sessions: dict[
            str,
            STTSession
        ] = {}

        self.worker = STTWorker(
            max_queue_size=max_queue_size
        )

        self.worker.on_transcript = (
            self._handle_transcript
        )

        self.transcript_callback: (
            TranscriptCallback | None
        ) = None

        self.running = False

    async def start(self):
        """
        Start the STT manager.
        """

        if self.running:
            return

        self.running = True

        await self.worker.start()

        print("STT Manager started.")

    async def stop(self):
        """
        Stop the STT manager gracefully.
        """

        if not self.running:
            return

        self.running = False

        await self.worker.stop()

        for session in list(
            self.sessions.values()
        ):
            session.close()

        self.sessions.clear()

        print("STT Manager stopped.")

    def create_session(
        self,
        session_id: str,
        language: str | None = None,
        sample_rate: int | None = None,
    ) -> STTSession:
        """
        Create one STT session for one call.
        """

        if session_id in self.sessions:
            raise ValueError(
                f"Session already exists: "
                f"{session_id}"
            )

        session = STTSession(
            session_id=session_id,
            sample_rate=(
                sample_rate
                or self.sample_rate
            ),
            language=(
                language
                if language is not None
                else self.language
            ),
        )

        self.sessions[
            session_id
        ] = session

        self.worker.register_session(
            session
        )

        return session

    async def submit_audio(
        self,
        session_id: str,
        audio_chunk: bytes,
    ):
        """
        Submit WebSocket audio for processing.
        """

        if not self.running:
            raise RuntimeError(
                "STT Manager is not running."
            )

        if session_id not in self.sessions:
            raise KeyError(
                f"Session not found: "
                f"{session_id}"
            )

        await self.worker.submit_audio(
            session_id,
            audio_chunk,
        )

    def set_language(
        self,
        session_id: str,
        language: str | None,
    ):
        """
        Change language for a call.
        """

        session = self._get_session(
            session_id
        )

        session.set_language(
            language
        )

    def end_session(
        self,
        session_id: str,
    ):
        """
        End and clean up one call.
        """

        session = self.sessions.pop(
            session_id,
            None,
        )

        if session is None:
            return

        self.worker.unregister_session(
            session_id
        )

        session.close()

    def get_session(
        self,
        session_id: str,
    ) -> STTSession:

        return self._get_session(
            session_id
        )

    def active_sessions(self) -> int:
        return len(self.sessions)

    def queue_size(self) -> int:
        return self.worker.queue_size()

    def set_transcript_callback(
        self,
        callback: TranscriptCallback,
    ):
        """
        Register callback for completed transcripts.

        Example:

            async def on_transcript(
                session_id,
                result,
            ):
                print(result.text)
        """

        self.transcript_callback = callback

    async def _handle_transcript(
        self,
        result: TranscriptResult,
        session_id: str,
    ):
        """
        Forward transcript to the conversation engine.
        """

        if self.transcript_callback:

            await self.transcript_callback(
                session_id,
                result,
            )

    def _get_session(
        self,
        session_id: str,
    ) -> STTSession:

        session = self.sessions.get(
            session_id
        )

        if session is None:
            raise KeyError(
                f"STT session not found: "
                f"{session_id}"
            )

        return session
