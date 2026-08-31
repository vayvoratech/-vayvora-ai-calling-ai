from src.voice.stt.core.manager import STTManager
from src.voice.stt.common.schemas import TranscriptResult


class STTWebSocketHandler:
    """
    Adapter between the Voice Session WebSocket
    and the STTManager.

    The WebSocket layer should call these methods.

    Lifecycle:

        start_call()
            â†“
        receive_audio()
            â†“
        receive_audio()
            â†“
        ...
            â†“
        end_call()
    """

    def __init__(
        self,
        stt_manager: STTManager,
    ):
        self.stt_manager = stt_manager

    async def start_call(
        self,
        session_id: str,
        language: str | None = "en",
        sample_rate: int = 16000,
    ):
        """
        Start STT for one voice call.
        """

        self.stt_manager.create_session(
            session_id=session_id,
            language=language,
            sample_rate=sample_rate,
        )

    async def receive_audio(
        self,
        session_id: str,
        audio_bytes: bytes,
    ):
        """
        Pass incoming WebSocket audio to STT.
        """

        await self.stt_manager.submit_audio(
            session_id=session_id,
            audio_chunk=audio_bytes,
        )

    async def end_call(
        self,
        session_id: str,
    ):
        """
        End STT for the call.
        """

        self.stt_manager.end_session(
            session_id
        )
