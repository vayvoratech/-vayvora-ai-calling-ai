import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from src.voice.stt.common.schemas import TranscriptResult
from src.voice.stt.core.session import STTSession


@dataclass
class AudioTask:
    """
    Audio belonging to one STT session.
    """

    session_id: str
    audio_chunk: bytes


TranscriptCallback = Callable[
    [TranscriptResult, str],
    Awaitable[None],
]


class STTWorker:
    """
    Asynchronous STT worker.

    WebSocket/audio reception stays on the asyncio
    event loop.

    CPU-heavy STT inference is executed in a
    background thread.
    """

    def __init__(
        self,
        max_queue_size: int = 100,
    ):
        self.queue: asyncio.Queue[
            AudioTask
        ] = asyncio.Queue(
            maxsize=max_queue_size
        )

        self.sessions: dict[
            str,
            STTSession,
        ] = {}

        self.running = False

        self.worker_task: asyncio.Task | None = None

        self.on_transcript: (
            TranscriptCallback | None
        ) = None

    # --------------------------------------------------
    # SESSION MANAGEMENT
    # --------------------------------------------------

    def register_session(
        self,
        session: STTSession,
    ):
        """
        Register an active STT session.
        """

        self.sessions[
            session.session_id
        ] = session

    def unregister_session(
        self,
        session_id: str,
    ):
        """
        Remove an STT session.
        """

        self.sessions.pop(
            session_id,
            None,
        )

    # --------------------------------------------------
    # AUDIO QUEUE
    # --------------------------------------------------

    async def submit_audio(
        self,
        session_id: str,
        audio_chunk: bytes,
    ):
        """
        Add audio to the processing queue.

        This method does NOT run Whisper.
        """

        if not audio_chunk:
            return

        if session_id not in self.sessions:
            raise KeyError(
                f"STT session not found: "
                f"{session_id}"
            )

        task = AudioTask(
            session_id=session_id,
            audio_chunk=audio_chunk,
        )

        await self.queue.put(task)

    # --------------------------------------------------
    # WORKER LIFECYCLE
    # --------------------------------------------------

    async def start(self):
        """
        Start background STT worker.
        """

        if self.running:
            return

        self.running = True

        self.worker_task = asyncio.create_task(
            self._run()
        )

    async def stop(self):
        """
        Stop the worker gracefully.
        """

        if not self.running:
            return

        self.running = False

        if self.worker_task:

            await self.worker_task

            self.worker_task = None

    # --------------------------------------------------
    # WORKER LOOP
    # --------------------------------------------------

    async def _run(self):
        """
        Main asynchronous worker loop.
        """

        while self.running:

            try:

                task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.5,
                )

            except asyncio.TimeoutError:
                continue

            try:

                session = self.sessions.get(
                    task.session_id
                )

                if session is None:
                    continue

                # --------------------------------------
                # IMPORTANT
                # --------------------------------------
                #
                # Whisper inference is CPU-heavy.
                #
                # Run the synchronous STT code
                # outside the asyncio event loop.
                #

                results = await asyncio.to_thread(
                    session.process_audio,
                    task.audio_chunk,
                )

                # --------------------------------------
                # SEND TRANSCRIPTS
                # --------------------------------------

                if self.on_transcript:

                    for result in results:

                        await self.on_transcript(
                            result,
                            task.session_id,
                        )

            except Exception as exc:

                print(
                    f"STT worker error "
                    f"[{task.session_id}]: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            finally:

                self.queue.task_done()

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    def queue_size(self) -> int:
        """
        Number of audio tasks waiting.
        """

        return self.queue.qsize()
