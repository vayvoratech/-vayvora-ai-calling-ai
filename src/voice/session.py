"""VoiceSession abstraction and deterministic mock audio transports for Pipecat orchestration.

Encapsulates active call lifecycle, bridges ConversationState to VoicePipeline,
and provides zero-hardware mock audio input and output sinks for automated testing.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple

from src.audio.schemas import AudioChunk, STTResult, TTSResult, VADResult
from src.core.errors import VoiceSessionError
from src.logging import get_logger
from src.state.models import ConversationState
from src.voice.context import TurnLifecycleState, VoiceDiagnostics
from src.voice.pipeline import VoicePipeline

logger = get_logger("voice.session")


# =============================================================================
# Mock Audio Transports & Sinks
# =============================================================================

class MockAudioInput:
    """Deterministic audio input queue simulating microphone, file, or packet stream."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.queue: List[AudioChunk] = []

    def feed_chunk(self, chunk: AudioChunk) -> None:
        """Enqueue an AudioChunk for pipeline consumption."""
        self.queue.append(chunk)

    def feed_pcm16_bytes(self, pcm_bytes: bytes, duration: float = 0.1) -> None:
        """Helper to enqueue raw PCM16 bytes as an AudioChunk."""
        chunk = AudioChunk(
            data=pcm_bytes,
            sample_rate=self.sample_rate,
            channels=1,
            format="pcm16",
            duration=duration,
        )
        self.feed_chunk(chunk)

    def read_chunk(self) -> Optional[AudioChunk]:
        """Fetch the next queued chunk, or None if buffer is empty."""
        return self.queue.pop(0) if self.queue else None

    def has_chunks(self) -> bool:
        """Check if unconsumed audio chunks remain in buffer."""
        return len(self.queue) > 0

    def clear(self) -> None:
        """Purge pending audio chunks."""
        self.queue.clear()


class MockAudioOutput:
    """Mock audio sink capturing synthesized TTS output frames for assertions."""

    def __init__(self) -> None:
        self.written_chunks: List[bytes] = []
        self.cleared_count: int = 0

    def write(self, data: bytes) -> None:
        """Append an audio byte payload to the sink buffer."""
        self.written_chunks.append(data)

    def clear(self) -> None:
        """Flush queued assistant audio immediately (called on barge-in)."""
        if self.written_chunks:
            self.cleared_count += len(self.written_chunks)
            self.written_chunks.clear()
            logger.info("MockAudioOutput: buffer flushed upon interruption.")

    def get_audio_data(self) -> bytes:
        """Concatenate all recorded audio bytes into a single buffer."""
        return b"".join(self.written_chunks)

    @property
    def total_bytes_written(self) -> int:
        """Total byte length of all active audio frames."""
        return sum(len(c) for c in self.written_chunks)


class MockPipecatTransport:
    """Simulated bidirectional Pipecat transport bridging input and output queues."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.input_sink = MockAudioInput(sample_rate=sample_rate)
        self.output_sink = MockAudioOutput()
        self.is_connected: bool = True

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False


# =============================================================================
# Voice Session
# =============================================================================

class VoiceSession:
    """Real-time voice session coordinating ConversationState and VoicePipeline."""

    def __init__(
        self,
        session_id: str,
        conversation_state: ConversationState,
        pipeline: VoicePipeline,
        audio_input: Optional[MockAudioInput] = None,
        audio_output: Optional[MockAudioOutput] = None,
    ) -> None:
        self.session_id = session_id
        self.conversation_state = conversation_state
        self.pipeline = pipeline
        self.audio_input = audio_input or MockAudioInput()
        self.audio_output = audio_output or (pipeline.output_sink or MockAudioOutput())
        self.pipeline.output_sink = self.audio_output

        # Timing and active status
        self.start_timestamp: float = time.time()
        self.end_timestamp: Optional[float] = None
        self.is_active: bool = False

    # -------------------------------------------------------------------------
    # State Inspection Properties
    # -------------------------------------------------------------------------

    @property
    def speech_state(self) -> str:
        """Current caller speech detection state."""
        if self.pipeline.state == TurnLifecycleState.CAPTURING:
            return "CAPTURING"
        elif self.pipeline.state == TurnLifecycleState.SPEECH_DETECTED:
            return "SPEECH_DETECTED"
        return "SILENCE"

    @property
    def agent_speaking_state(self) -> bool:
        """True if the voice pipeline is actively speaking assistant response."""
        return self.pipeline.state == TurnLifecycleState.SPEAKING

    @property
    def interruption_state(self) -> bool:
        """True if caller has triggered barge-in during this session."""
        return (
            self.pipeline.state == TurnLifecycleState.INTERRUPTED
            or self.pipeline.diagnostics.interruption_count > 0
        )

    @property
    def diagnostics(self) -> VoiceDiagnostics:
        """Access real-time pipeline latency and counter metrics."""
        return self.pipeline.diagnostics

    # -------------------------------------------------------------------------
    # Lifecycle Controls
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the voice session (for inbound calls, waits in LISTENING state for caller)."""
        self.is_active = True
        self.start_timestamp = time.time()
        self.pipeline.state = TurnLifecycleState.LISTENING
        logger.info("VoiceSession %s started.", self.session_id)

    async def start_outbound(self, engine: Optional[Any] = None) -> Optional[Any]:
        """Start an outbound voice session, proactively generating and voicing the agent opening.

        Leaves the session active and in LISTENING state waiting for caller response.
        """
        self.start()
        conv_engine = engine or getattr(getattr(self.pipeline, "conversation", None), "engine", None)
        if conv_engine and hasattr(conv_engine, "start_outbound_conversation"):
            turn_result = await conv_engine.start_outbound_conversation(self.conversation_state)
            # Synthesize speech to audio_output if TTS provider is present
            tts_adapter = getattr(self.pipeline, "tts", None)
            tts_provider = getattr(tts_adapter, "provider", None) or tts_adapter
            if tts_provider and self.audio_output and hasattr(tts_provider, "synthesize"):
                try:
                    tts_res = await tts_provider.synthesize(turn_result.response_text)
                    self.audio_output.write(tts_res.audio_data)
                except Exception as tts_err:
                    logger.warning("Could not synthesize outbound opening speech: %s", tts_err)
            self.pipeline.state = TurnLifecycleState.LISTENING
            return turn_result
        return None

    def stop(self, reason: str = "caller_hung_up") -> None:
        """Gracefully stop the voice session."""
        self.is_active = False
        self.end_timestamp = time.time()
        self.pipeline.state = TurnLifecycleState.COMPLETED
        self.pipeline.cancellation_token.cancel()
        self.conversation_state.conversation_active = False
        self.conversation_state.termination_reason = reason
        logger.info("VoiceSession %s stopped (Reason: %s).", self.session_id, reason)

    # -------------------------------------------------------------------------
    # Frame Processing
    # -------------------------------------------------------------------------

    async def process_audio_chunk(
        self, chunk: AudioChunk
    ) -> Tuple[VADResult, Optional[STTResult], Optional[TTSResult]]:
        """Feed a single AudioChunk into the pipeline."""
        if not self.is_active:
            raise VoiceSessionError(f"Cannot process audio: session {self.session_id} is inactive.")

        vad_res, stt_res, tts_res = await self.pipeline.process_audio_chunk(
            chunk=chunk,
            conversation_state=self.conversation_state,
        )

        if not self.conversation_state.conversation_active and self.is_active:
            self.stop(reason=self.conversation_state.termination_reason or "completed")

        return vad_res, stt_res, tts_res

    async def process_all_queued_input(self) -> List[Tuple[VADResult, Optional[STTResult], Optional[TTSResult]]]:
        """Drain and process all pending audio chunks in audio_input buffer."""
        results = []
        while self.audio_input.has_chunks() and self.is_active:
            chunk = self.audio_input.read_chunk()
            if chunk:
                res = await self.process_audio_chunk(chunk)
                results.append(res)
        return results

    def get_diagnostics(self) -> Dict[str, Any]:
        """Serialize session telemetry and timing metrics."""
        data = self.diagnostics.to_dict()
        data.update({
            "is_active": self.is_active,
            "speech_state": self.speech_state,
            "agent_speaking_state": self.agent_speaking_state,
            "interruption_state": self.interruption_state,
            "total_duration": (
                round((self.end_timestamp or time.time()) - self.start_timestamp, 4)
            ),
        })
        return data
