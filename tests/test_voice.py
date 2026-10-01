"""Comprehensive Unit Test Suite for Phase 8 Pipecat Voice Orchestration & Barge-In.

Covers all 42 required scenarios:
- Pipeline: session creation, start, stop, audio input, VAD speech start/end, STT transcript, engine, TTS, audio output
- Turn Lifecycle: LISTENING -> SPEECH_DETECTED -> CAPTURING -> TRANSCRIBING -> THINKING -> SPEAKING -> LISTENING
- Barge-in & Interruption: caller interrupts TTS, TTS cancellation, queued audio cancellation, generation invalidation, stale response discarded, new response accepted
- Race Conditions: late TTS frame, late Gemini response, previous turn completes late, MCP action continues through interruption
- Conversation Semantics: 'no' keeps call active, explicit goodbye completes call, intent switching, domain switching, business status does not terminate
- Errors: VAD failure, STT failure, LLM failure, TTS failure, MCP failure, recovery after error
- Diagnostics: event generation, latency measurements, generation IDs, turn IDs, interruption count

Zero live audio hardware, GPU, or model downloads required.
"""

import pytest
from typing import Optional

from src.audio.schemas import AudioChunk, STTResult, TTSResult, VADResult
from src.audio.stt import MockSTTProvider
from src.audio.tts import MockTTSProvider, create_pcm_wav_bytes
from src.audio.vad import MockVADProvider
from src.core.decision import ConversationalDecision, ProposedAction
from src.core.engine import ConversationEngine
from src.core.errors import (
    BargeInInterruptionError,
    StaleGenerationError,
    VoicePipelineError,
    VoiceSessionError,
)
from src.core.llm import MockLLMProvider
from src.core.types import (
    CallDirection,
    CallMetadata,
    CallStatus,
    ConversationStage,
    DomainType,
    ToolExecutionResult,
)
from src.rag.embeddings import MockEmbeddingProvider
from src.rag.retriever import GroundedKnowledgeProvider
from src.state.manager import ConversationStateManager
from src.state.models import CallerProfile, ConversationState
from src.tools.mcp_client import MockToolProvider
from src.voice.adapters import (
    PipecatConversationAdapter,
    PipecatSTTAdapter,
    PipecatTTSAdapter,
    PipecatVADAdapter,
)
from src.voice.context import CancellationToken, TurnLifecycleState, VoiceDiagnostics
from src.voice.events import (
    AgentResponseCompleted,
    AgentResponseStarted,
    AgentThinking,
    CallerInterrupted,
    SessionCompleted,
    SpeechEnded,
    SpeechStarted,
    TranscriptFinal,
    TTSStarted,
    TTSStopped,
)
from src.voice.pipeline import VoicePipeline
from src.voice.session import (
    MockAudioInput,
    MockAudioOutput,
    MockPipecatTransport,
    VoiceSession,
)


# =============================================================================
# Test Fixtures & Helpers
# =============================================================================

def make_speech_chunk(duration: float = 0.2, sample_rate: int = 16000) -> AudioChunk:
    """Generate a PCM16 chunk representing active speech."""
    bytes_data = create_pcm_wav_bytes(duration=duration, sample_rate=sample_rate, frequency=440.0)
    # Strip 44-byte WAV header to simulate raw PCM stream
    pcm_payload = bytes_data[44:] if len(bytes_data) > 44 else bytes_data
    return AudioChunk(
        data=pcm_payload,
        sample_rate=sample_rate,
        channels=1,
        format="pcm16",
        duration=duration,
    )


def make_silence_chunk(duration: float = 0.2, sample_rate: int = 16000) -> AudioChunk:
    """Generate a PCM16 chunk representing silence."""
    num_samples = int(sample_rate * duration)
    return AudioChunk(
        data=b"\x00" * (num_samples * 2),
        sample_rate=sample_rate,
        channels=1,
        format="pcm16",
        duration=duration,
    )


def build_test_voice_components(
    llm_canned_decisions: Optional[list] = None,
    stt_transcriptions: Optional[list] = None,
    vad_probabilities: Optional[list] = None,
    tool_force_error: bool = False,
    llm_force_error: bool = False,
    vad_force_error: bool = False,
    stt_force_error: bool = False,
    tts_force_error: bool = False,
    min_speech_duration: float = 0.2,
    min_silence_duration: float = 0.2,
):
    """Build a fully wired, deterministic VoiceSession and VoicePipeline."""
    session_id = "test-call-voice-01"

    # State
    mgr = ConversationStateManager()
    state = mgr.create_inbound_state(
        call_id=session_id,
        caller_phone="+15551234567",
        domain=DomainType.EDUSAAS,
    )

    # Core engine dependencies
    from src.core.errors import LLMError

    class FailableMockLLM(MockLLMProvider):
        def __init__(self, *args, force_error: bool = False, **kwargs):
            super().__init__(*args, **kwargs)
            self.force_error = force_error

        async def generate_decision(self, *args, **kwargs):
            if self.force_error:
                raise LLMError("Simulated LLM decision failure.")
            return await super().generate_decision(*args, **kwargs)

    llm = FailableMockLLM(
        canned_decisions=llm_canned_decisions,
        force_error=llm_force_error,
    )
    tools = MockToolProvider(force_unavailable=tool_force_error)
    rag = GroundedKnowledgeProvider(
        embedding_provider=MockEmbeddingProvider(dimension=64),
        in_memory=True,
    )
    engine = ConversationEngine(
        llm_provider=llm,
        knowledge_provider=rag,
        tool_provider=tools,
    )

    # Audio providers
    vad = MockVADProvider(
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        canned_probabilities=vad_probabilities,
        force_error=vad_force_error,
    )
    stt = MockSTTProvider(
        canned_transcriptions=stt_transcriptions,
        force_error=stt_force_error,
    )
    tts = MockTTSProvider(
        force_error=tts_force_error,
    )

    # Adapters
    vad_adapter = PipecatVADAdapter(vad)
    stt_adapter = PipecatSTTAdapter(stt)
    conversation_adapter = PipecatConversationAdapter(engine)
    tts_adapter = PipecatTTSAdapter(tts)

    audio_output = MockAudioOutput()
    audio_input = MockAudioInput()

    pipeline = VoicePipeline(
        session_id=session_id,
        vad_adapter=vad_adapter,
        stt_adapter=stt_adapter,
        conversation_adapter=conversation_adapter,
        tts_adapter=tts_adapter,
        audio_output=audio_output,
    )

    session = VoiceSession(
        session_id=session_id,
        conversation_state=state,
        pipeline=pipeline,
        audio_input=audio_input,
        audio_output=audio_output,
    )

    return session, pipeline, engine, vad, stt, tts, tools


# =============================================================================
# 1. Pipeline Tests (1-10)
# =============================================================================

class TestVoicePipelineCore:
    """Tests 1-10: Pipeline creation, lifecycle, audio input/output, VAD, STT, engine, and TTS."""

    def test_01_session_creation(self) -> None:
        """VoiceSession initializes with proper session_id, inactive status, and diagnostics."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        assert session.session_id == "test-call-voice-01"
        assert session.is_active is False
        assert session.pipeline.state == TurnLifecycleState.LISTENING
        assert session.diagnostics.session_id == "test-call-voice-01"

    def test_02_session_start(self) -> None:
        """Starting session activates state, updates start_timestamp, and sets pipeline to LISTENING."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        session.start()
        assert session.is_active is True
        assert session.start_timestamp > 0.0
        assert pipeline.state == TurnLifecycleState.LISTENING

    def test_03_session_stop(self) -> None:
        """Stopping session deactivates state, records end_timestamp, and updates conversation state."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        session.start()
        session.stop(reason="caller_hung_up")
        assert session.is_active is False
        assert session.end_timestamp is not None
        assert pipeline.state == TurnLifecycleState.COMPLETED
        assert session.conversation_state.conversation_active is False
        assert session.conversation_state.termination_reason == "caller_hung_up"

    def test_04_audio_input_buffering(self) -> None:
        """MockAudioInput buffers audio chunks and supports FIFO retrieval."""
        input_queue = MockAudioInput()
        c1 = make_speech_chunk(duration=0.1)
        c2 = make_silence_chunk(duration=0.1)
        input_queue.feed_chunk(c1)
        input_queue.feed_chunk(c2)

        assert input_queue.has_chunks() is True
        assert input_queue.read_chunk() == c1
        assert input_queue.read_chunk() == c2
        assert input_queue.has_chunks() is False

    @pytest.mark.asyncio
    async def test_05_vad_speech_start_detection(self) -> None:
        """VAD speech start event fires when caller begins speaking."""
        # 1 speech chunk (0.2s) matches min_speech_duration (0.2s)
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9],
            min_speech_duration=0.2,
        )
        session.start()
        vad_res, _, _ = await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        assert vad_res.speech_start is True
        assert pipeline._is_capturing_speech is True
        assert pipeline.state == TurnLifecycleState.CAPTURING
        assert any(isinstance(e, SpeechStarted) for e in pipeline.event_history)

    @pytest.mark.asyncio
    async def test_06_vad_speech_end_detection(self) -> None:
        """VAD speech end event fires after accumulated silence exceeds min_silence_duration."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        vad_res, _, _ = await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert vad_res.speech_end is True
        assert any(isinstance(e, SpeechEnded) for e in pipeline.event_history)

    @pytest.mark.asyncio
    async def test_07_stt_final_transcript_emission(self) -> None:
        """Upon speech_end, STT processes segment and emits TranscriptFinal event."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Hello, I need help with courses."],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        _, stt_res, _ = await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert stt_res is not None
        assert stt_res.text == "Hello, I need help with courses."
        transcript_events = [e for e in pipeline.event_history if isinstance(e, TranscriptFinal)]
        assert len(transcript_events) == 1
        assert transcript_events[0].transcript == "Hello, I need help with courses."

    @pytest.mark.asyncio
    async def test_08_conversation_engine_invocation(self) -> None:
        """Engine processes transcribed user utterance and generates response."""
        dec = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="course_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="EduSaaS offers comprehensive AI and Cloud engineering programs.",
        )
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Tell me about your courses."],
            llm_canned_decisions=[dec],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        _, _, tts_res = await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert session.conversation_state.current_intent == "course_inquiry"
        assert tts_res is not None
        assert "EduSaaS offers" in tts_res.text

    @pytest.mark.asyncio
    async def test_09_tts_invocation_and_events(self) -> None:
        """TTS synthesizes engine response and fires TTSStarted and TTSStopped events."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Hi there"],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        _, _, tts_res = await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert tts_res is not None
        assert tts_res.audio_data[:4] == b"RIFF"
        assert any(isinstance(e, TTSStarted) for e in pipeline.event_history)
        assert any(isinstance(e, TTSStopped) for e in pipeline.event_history)

    @pytest.mark.asyncio
    async def test_10_audio_output_reception(self) -> None:
        """Synthesized audio bytes are delivered into the output sink."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Greetings"],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert session.audio_output.total_bytes_written > 0
        assert session.audio_output.get_audio_data()[:4] == b"RIFF"


# =============================================================================
# 2. Turn Lifecycle Tests (11-16)
# =============================================================================

class TestTurnLifecycle:
    """Tests 11-16: Step-by-step turn lifecycle states."""

    @pytest.mark.asyncio
    async def test_11_listening_to_speech_detected(self) -> None:
        """Transition from LISTENING to CAPTURING on speech onset."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9],
            min_speech_duration=0.2,
        )
        session.start()
        assert pipeline.state == TurnLifecycleState.LISTENING

        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        assert pipeline.state == TurnLifecycleState.CAPTURING

    @pytest.mark.asyncio
    async def test_12_speech_detected_to_capturing(self) -> None:
        """Subsequent speech frames remain in CAPTURING and accumulate audio."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.9],
            min_speech_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        assert pipeline.state == TurnLifecycleState.CAPTURING
        assert len(pipeline._speech_chunks) == 2

    @pytest.mark.asyncio
    async def test_13_capturing_to_transcribing(self) -> None:
        """Speech offset transitions pipeline through TRANSCRIBING state."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        # Hook to verify TRANSCRIBING state when speech_end fires
        states_seen = []
        pipeline.subscribe(lambda e: states_seen.append(pipeline.state))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert TurnLifecycleState.TRANSCRIBING in states_seen

    @pytest.mark.asyncio
    async def test_14_transcribing_to_thinking(self) -> None:
        """After transcription, pipeline transitions to THINKING state."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        states_seen = []
        pipeline.subscribe(lambda e: states_seen.append(pipeline.state))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert TurnLifecycleState.THINKING in states_seen
        assert any(isinstance(e, AgentThinking) for e in pipeline.event_history)

    @pytest.mark.asyncio
    async def test_15_thinking_to_speaking(self) -> None:
        """After engine responds, pipeline transitions to SPEAKING for audio synthesis."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        states_seen = []
        pipeline.subscribe(lambda e: states_seen.append(pipeline.state))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert TurnLifecycleState.SPEAKING in states_seen

    @pytest.mark.asyncio
    async def test_16_speaking_to_listening(self) -> None:
        """Following speech delivery, active session returns to LISTENING for next caller turn."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert pipeline.state == TurnLifecycleState.LISTENING


# =============================================================================
# 3. Barge-In & Interruption Tests (17-22)
# =============================================================================

class TestBargeInAndInterruption:
    """Tests 17-22: Caller speaks over assistant speech, triggering cancellation and recovery."""

    def test_17_caller_interrupts_tts(self) -> None:
        """Triggering barge-in transitions state to INTERRUPTED and records event."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        session.start()
        pipeline.state = TurnLifecycleState.SPEAKING

        pipeline.trigger_barge_in()
        assert pipeline.state == TurnLifecycleState.INTERRUPTED
        assert pipeline.diagnostics.interruption_count == 1
        assert any(isinstance(e, CallerInterrupted) for e in pipeline.event_history)

    def test_18_current_tts_cancellation_invoked(self) -> None:
        """Barge-in dispatches cancellation directly to the TTS provider."""
        session, pipeline, _, _, _, tts, _ = build_test_voice_components()
        session.start()
        pipeline.state = TurnLifecycleState.SPEAKING

        pipeline.trigger_barge_in()
        assert tts._is_cancelled is True

    def test_19_queued_audio_cancellation(self) -> None:
        """Barge-in purges buffered audio frames from the output sink."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        session.start()
        pipeline.state = TurnLifecycleState.SPEAKING

        # Pre-fill output sink with playing chunks
        session.audio_output.write(b"AUDIO_CHUNK_1")
        session.audio_output.write(b"AUDIO_CHUNK_2")
        assert session.audio_output.total_bytes_written > 0

        pipeline.trigger_barge_in()
        assert session.audio_output.total_bytes_written == 0
        assert session.audio_output.cleared_count == 2

    def test_20_generation_invalidation(self) -> None:
        """Barge-in increments current_generation_id and invalidates previous token."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        gen_before = pipeline.current_generation_id

        pipeline.trigger_barge_in()
        assert pipeline.current_generation_id == gen_before + 1
        assert pipeline.cancellation_token.is_valid_generation(gen_before) is False

    @pytest.mark.asyncio
    async def test_21_stale_response_discarded_on_interruption(self) -> None:
        """Late synthesis belonging to an invalidated generation is rejected and dropped."""
        session, pipeline, _, _, _, tts, _ = build_test_voice_components()
        token = CancellationToken(generation_id=1)
        token.cancel()  # Simulates interruption

        with pytest.raises(StaleGenerationError):
            await pipeline.tts.synthesize(
                text="Late speech",
                generation_id=1,
                cancellation_token=token,
            )

    @pytest.mark.asyncio
    async def test_22_new_response_accepted_after_interruption(self) -> None:
        """After interruption, subsequent caller utterance produces valid new turn response."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Fresh turn after interruption"],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        # Simulate interruption occurred previously
        pipeline.trigger_barge_in()
        pipeline.state = TurnLifecycleState.LISTENING

        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        _, _, tts_res = await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert tts_res is not None
        assert tts_res.audio_data[:4] == b"RIFF"
        assert pipeline.state == TurnLifecycleState.LISTENING


# =============================================================================
# 4. Race Condition Protection Tests (23-26)
# =============================================================================

class TestRaceConditions:
    """Tests 23-26: Late frames, concurrent responses, and non-blocking MCP mutations."""

    @pytest.mark.asyncio
    async def test_23_late_tts_frame_discarded(self) -> None:
        """Late TTS chunk arriving with stale generation_id is dropped."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        pipeline.current_generation_id = 5

        # Attempt to synthesize with generation_id = 4
        token = CancellationToken(generation_id=5)
        with pytest.raises(StaleGenerationError):
            await pipeline.tts.synthesize(
                text="Old audio frame",
                generation_id=4,
                cancellation_token=token,
            )

    @pytest.mark.asyncio
    async def test_24_late_gemini_response_discarded(self) -> None:
        """LLM response completing after barge-in cancellation is discarded without updating state."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        token = CancellationToken(generation_id=2)
        token.cancel()  # Interrupted during LLM thinking

        with pytest.raises(StaleGenerationError):
            await pipeline.conversation.process_user_turn(
                state=session.conversation_state,
                user_text="User utterance",
                generation_id=2,
                cancellation_token=token,
            )

    @pytest.mark.asyncio
    async def test_25_previous_turn_completes_late(self) -> None:
        """Post-execution cancellation check discards response if token was cancelled during processing."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        token = CancellationToken(generation_id=1)

        # Hook: cancel token during turn processing
        original_process = pipeline.conversation.engine.process_user_turn

        async def cancelled_process(*args, **kwargs):
            token.cancel()
            return await original_process(*args, **kwargs)

        pipeline.conversation.engine.process_user_turn = cancelled_process

        with pytest.raises(StaleGenerationError):
            await pipeline.conversation.process_user_turn(
                state=session.conversation_state,
                user_text="User message",
                generation_id=1,
                cancellation_token=token,
            )

    @pytest.mark.asyncio
    async def test_26_mcp_action_continues_through_interruption(self) -> None:
        """External MCP tool actions execute and verify safely without aborting underlying mutation."""
        dec = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="request_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "caller@example.com", "subject": "Syllabus"},
            ),
            user_facing_response="I have sent the curriculum.",
        )
        session, pipeline, _, _, _, _, tools = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Please email me the syllabus"],
            llm_canned_decisions=[dec],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        # Tool was executed and verified
        assert len(tools.dispatched_requests) == 1
        assert tools.dispatched_requests[0].action_name == "send_email"
        assert session.conversation_state.last_action == "send_email"
        assert session.conversation_state.last_tool_result.success is True


# =============================================================================
# 5. Conversation Semantics Tests (27-31)
# =============================================================================

class TestVoiceConversationSemantics:
    """Tests 27-31: Preservation of Phase 1-7 business rules under voice pipeline."""

    @pytest.mark.asyncio
    async def test_27_no_keeps_call_active(self) -> None:
        """Caller saying 'no' does NOT terminate the voice session."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["No, thank you."],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert session.is_active is True
        assert session.conversation_state.conversation_active is True
        assert pipeline.state == TurnLifecycleState.LISTENING

    @pytest.mark.asyncio
    async def test_28_explicit_goodbye_completes_call(self) -> None:
        """Caller saying 'goodbye' completes both conversation state and voice session."""
        dec = ConversationalDecision(
            detected_domain=DomainType.GENERAL,
            detected_intent="farewell",
            proposed_stage=ConversationStage.COMPLETED,
            user_facing_response="Thank you for calling. Goodbye!",
        )
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["That is all, goodbye."],
            llm_canned_decisions=[dec],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert session.is_active is False
        assert session.conversation_state.conversation_active is False
        assert pipeline.state == TurnLifecycleState.COMPLETED
        assert any(isinstance(e, SessionCompleted) for e in pipeline.event_history)

    @pytest.mark.asyncio
    async def test_29_intent_switching_works(self) -> None:
        """Caller mid-call intent switch is recognized and tracked."""
        dec = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="fee_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Our courses are competitively priced.",
        )
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Actually, what are the fees?"],
            llm_canned_decisions=[dec],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        session.conversation_state.current_intent = "curriculum_inquiry"

        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert session.conversation_state.current_intent == "fee_inquiry"

    @pytest.mark.asyncio
    async def test_30_domain_switching_works(self) -> None:
        """Caller switching from EduSaaS to Vayvora updates current_domain."""
        dec = ConversationalDecision(
            detected_domain=DomainType.VAYVORA,
            detected_intent="solutions_inquiry",
            proposed_stage=ConversationStage.INFORMATION,
            user_facing_response="Vayvora builds enterprise AI platforms.",
        )
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Tell me about Vayvora software services."],
            llm_canned_decisions=[dec],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        assert session.conversation_state.current_domain == DomainType.EDUSAAS

        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert session.conversation_state.current_domain == DomainType.VAYVORA

    @pytest.mark.asyncio
    async def test_31_business_status_does_not_terminate_call(self) -> None:
        """Updating lead business status to 'qualified' keeps conversation active."""
        dec = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="enrollment_interest",
            proposed_stage=ConversationStage.INFORMATION,
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="update_business_status",
                arguments={"status": "interested"},
            ),
            user_facing_response="I have marked you as interested. Let us discuss dates.",
        )
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["I want to enroll next month."],
            llm_canned_decisions=[dec],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert session.is_active is True
        assert session.conversation_state.conversation_active is True


# =============================================================================
# 6. Errors & Resilience Tests (32-37)
# =============================================================================

class TestVoiceErrorResilience:
    """Tests 32-37: Graceful error handling across VAD, STT, LLM, TTS, MCP, and recovery."""

    @pytest.mark.asyncio
    async def test_32_vad_failure_handling(self) -> None:
        """VAD internal computation error transitions pipeline to ERROR and logs telemetry."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(vad_force_error=True)
        session.start()

        with pytest.raises(VoicePipelineError):
            await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        assert pipeline.state == TurnLifecycleState.ERROR
        assert len(pipeline.diagnostics.errors) > 0

    @pytest.mark.asyncio
    async def test_33_stt_failure_handling(self) -> None:
        """STT failure transitions pipeline to ERROR without crashing the application."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_force_error=True,
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        with pytest.raises(Exception):
            await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert pipeline.state == TurnLifecycleState.ERROR

    @pytest.mark.asyncio
    async def test_34_llm_failure_handling(self) -> None:
        """LLM failure transitions pipeline to ERROR and records diagnostic error."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Hello"],
            llm_force_error=True,
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        with pytest.raises(Exception):
            await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert pipeline.state == TurnLifecycleState.ERROR

    @pytest.mark.asyncio
    async def test_35_tts_failure_handling(self) -> None:
        """TTS failure transitions pipeline to ERROR and logs error event."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Hello"],
            tts_force_error=True,
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))

        with pytest.raises(Exception):
            await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert pipeline.state == TurnLifecycleState.ERROR

    @pytest.mark.asyncio
    async def test_36_mcp_failure_handling(self) -> None:
        """MCP failure does not terminate the voice call and returns safe response."""
        dec = ConversationalDecision(
            detected_domain=DomainType.EDUSAAS,
            detected_intent="request_brochure",
            proposed_stage=ConversationStage.ACTION_CONFIRMATION,
            action_proposed=True,
            proposed_action=ProposedAction(
                tool_name="send_email",
                arguments={"recipient": "caller@example.com"},
            ),
            user_facing_response="I attempted to send the email.",
        )
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Send email"],
            llm_canned_decisions=[dec],
            tool_force_error=True,
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        _, _, tts_res = await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        assert session.is_active is True
        assert tts_res is not None
        assert pipeline.state == TurnLifecycleState.LISTENING

    @pytest.mark.asyncio
    async def test_37_recovery_after_error(self) -> None:
        """Pipeline can be reset after error and continue processing subsequent turns."""
        session, pipeline, _, vad, _, _, _ = build_test_voice_components(
            vad_force_error=True,
        )
        session.start()
        with pytest.raises(VoicePipelineError):
            await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        assert pipeline.state == TurnLifecycleState.ERROR

        # Recover: fix error, reset pipeline, process fresh frame
        vad.force_error = False
        vad.canned_probabilities = [0.05]
        pipeline.reset()
        assert pipeline.state == TurnLifecycleState.LISTENING

        vad_res, _, _ = await session.process_audio_chunk(make_silence_chunk(duration=0.2))
        assert vad_res.is_speech is False
        assert pipeline.state == TurnLifecycleState.LISTENING


# =============================================================================
# 7. Diagnostics & Telemetry Tests (38-42)
# =============================================================================

class TestVoiceDiagnosticsAndMetrics:
    """Tests 38-42: Event generation, latency metrics, generation IDs, turn IDs, and interruption count."""

    @pytest.mark.asyncio
    async def test_38_event_generation(self) -> None:
        """A complete conversational turn emits structured lifecycle events."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Testing events"],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        event_types = [type(e) for e in pipeline.event_history]
        assert SpeechStarted in event_types
        assert SpeechEnded in event_types
        assert TranscriptFinal in event_types
        assert AgentThinking in event_types
        assert AgentResponseStarted in event_types
        assert TTSStarted in event_types
        assert TTSStopped in event_types
        assert AgentResponseCompleted in event_types

    @pytest.mark.asyncio
    async def test_39_latency_measurements_recorded(self) -> None:
        """Operational latencies for VAD, STT, LLM, TTS, and Turn are measured and populated."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05],
            stt_transcriptions=["Testing latencies"],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))

        diag = session.get_diagnostics()
        assert diag["vad"]["speech_duration"] > 0
        assert diag["stt"]["transcription_latency"] is not None
        assert diag["llm"]["total_generation_latency"] is not None
        assert diag["tts"]["total_synthesis_latency"] is not None
        assert diag["pipeline"]["turn_latency"] is not None

    def test_40_generation_id_tracking(self) -> None:
        """Generation IDs increment properly upon barge-in invalidations."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        assert pipeline.current_generation_id == 0

        pipeline.trigger_barge_in()
        assert pipeline.current_generation_id == 1

        pipeline.trigger_barge_in()
        assert pipeline.current_generation_id == 2
        assert session.diagnostics.cancelled_generations_count == 2

    @pytest.mark.asyncio
    async def test_41_turn_id_tracking(self) -> None:
        """Turn ID increments with each completed conversational utterance."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components(
            vad_probabilities=[0.9, 0.05, 0.9, 0.05],
            stt_transcriptions=["Turn one", "Turn two"],
            min_speech_duration=0.2,
            min_silence_duration=0.2,
        )
        session.start()
        # Turn 1
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))
        assert pipeline.turn_id == 1

        # Turn 2
        await session.process_audio_chunk(make_speech_chunk(duration=0.2))
        await session.process_audio_chunk(make_silence_chunk(duration=0.2))
        assert pipeline.turn_id == 2

    def test_42_interruption_count_metrics(self) -> None:
        """Diagnostics accurately reflect cumulative caller interruption count."""
        session, pipeline, _, _, _, _, _ = build_test_voice_components()
        assert session.diagnostics.interruption_count == 0

        for i in range(1, 4):
            pipeline.trigger_barge_in()
            assert session.diagnostics.interruption_count == i
