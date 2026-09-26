"""Comprehensive Unit Test Suite for Phase 7 Local Audio Engines.

Tests deterministic VAD, STT, and TTS provider implementations, schemas,
temporal speech-silence state machines, streaming, cancellation, and test pipeline.
No external network calls or heavy neural model downloads required.
"""

import pytest
from typing import AsyncIterator

from src.audio.pipeline import AudioPipeline
from src.audio.schemas import AudioChunk, SpeechSegment, STTResult, TTSResult, VADResult
from src.audio.stt import MockSTTProvider
from src.audio.tts import MockTTSProvider, create_pcm_wav_bytes
from src.audio.vad import MockVADProvider
from src.config import Settings
from src.core.errors import (
    AudioPipelineError,
    InvalidAudioDataError,
    STTTranscriptionError,
    TTSSynthesisError,
    VADProcessingError,
)
from src.ui.bootstrap import bootstrap_workbench
from src.ui.service import ServiceComponents, WorkbenchService


# =============================================================================
# Helper Utilities
# =============================================================================

def make_pcm16_chunk(
    duration: float = 0.1,
    sample_rate: int = 16000,
    amplitude: int = 5000,
) -> AudioChunk:
    """Generate a valid PCM16 AudioChunk for unit testing."""
    num_samples = int(sample_rate * duration)
    # Sine wave or non-zero signal
    import numpy as np
    t = np.linspace(0, duration, num_samples, endpoint=False)
    data = (np.sin(2 * np.pi * 440.0 * t) * amplitude).astype(np.int16).tobytes()
    return AudioChunk(
        data=data,
        sample_rate=sample_rate,
        channels=1,
        format="pcm16",
        duration=duration,
    )


def make_silence_chunk(duration: float = 0.1, sample_rate: int = 16000) -> AudioChunk:
    """Generate a PCM16 silence AudioChunk (all zeros)."""
    num_samples = int(sample_rate * duration)
    data = b"\x00" * (num_samples * 2)
    return AudioChunk(
        data=data,
        sample_rate=sample_rate,
        channels=1,
        format="pcm16",
        duration=duration,
    )


# =============================================================================
# 1. Voice Activity Detection (VAD) Tests
# =============================================================================

class TestVADProvider:
    """Unit tests for VAD state machine, speech boundary detection, and edge handling."""

    def test_01_speech_detection(self) -> None:
        """Active speech frame is classified as speech with high probability."""
        vad = MockVADProvider(threshold=0.5, canned_probabilities=[0.9])
        chunk = make_pcm16_chunk(duration=0.1)
        res = vad.process_chunk(chunk)
        assert res.confidence == 0.9
        assert res.duration == 0.1

    def test_02_silence_detection(self) -> None:
        """Silence frame is classified as non-speech with low probability."""
        vad = MockVADProvider(threshold=0.5, canned_probabilities=[0.05])
        chunk = make_silence_chunk(duration=0.1)
        res = vad.process_chunk(chunk)
        assert not res.is_speech
        assert res.confidence == 0.05

    def test_03_speech_start_edge_detection(self) -> None:
        """Leading edge transition from silence to speech sets speech_start=True."""
        # min_speech_duration = 0.2s -> 2 chunks of 0.1s needed
        vad = MockVADProvider(
            threshold=0.5,
            min_speech_duration=0.2,
            canned_probabilities=[0.9, 0.9],
        )
        c1 = make_pcm16_chunk(duration=0.1)
        res1 = vad.process_chunk(c1)
        assert not res1.speech_start  # Not yet met 0.2s

        c2 = make_pcm16_chunk(duration=0.1)
        res2 = vad.process_chunk(c2)
        assert res2.speech_start
        assert res2.is_speech
        assert vad.is_speaking

    def test_04_speech_end_edge_detection(self) -> None:
        """Trailing edge transition from speech to silence sets speech_end=True."""
        # Establish speech first
        vad = MockVADProvider(
            threshold=0.5,
            min_speech_duration=0.1,
            min_silence_duration=0.2,
            canned_probabilities=[0.9, 0.1, 0.1],
        )
        res_speech = vad.process_chunk(make_pcm16_chunk(duration=0.1))
        assert res_speech.speech_start
        assert vad.is_speaking

        # Silence 1 (0.1s < 0.2s min_silence)
        res_sil1 = vad.process_chunk(make_silence_chunk(duration=0.1))
        assert not res_sil1.speech_end
        assert vad.is_speaking  # Still considered in speech block

        # Silence 2 (accumulated 0.2s >= 0.2s min_silence)
        res_sil2 = vad.process_chunk(make_silence_chunk(duration=0.1))
        assert res_sil2.speech_end
        assert not res_sil2.is_speech
        assert not vad.is_speaking

    def test_05_min_speech_duration_filters_noise_burst(self) -> None:
        """Brief noise burst shorter than min_speech_duration does not trigger speech_start."""
        vad = MockVADProvider(
            threshold=0.5,
            min_speech_duration=0.3,
            canned_probabilities=[0.9, 0.1],  # 0.1s noise followed by silence
        )
        r1 = vad.process_chunk(make_pcm16_chunk(duration=0.1))
        assert not r1.speech_start

        r2 = vad.process_chunk(make_silence_chunk(duration=0.1))
        assert not r2.speech_start
        assert not r2.speech_end
        assert not vad.is_speaking

    def test_06_min_silence_duration_prevents_premature_speech_end(self) -> None:
        """Brief pause (e.g. natural breathing) does not trigger speech_end."""
        vad = MockVADProvider(
            threshold=0.5,
            min_speech_duration=0.1,
            min_silence_duration=0.3,
            canned_probabilities=[0.9, 0.1, 0.9],
        )
        # Speech onset
        r1 = vad.process_chunk(make_pcm16_chunk(duration=0.1))
        assert r1.speech_start

        # Brief pause 0.1s
        r2 = vad.process_chunk(make_silence_chunk(duration=0.1))
        assert not r2.speech_end
        assert vad.is_speaking

        # Resumes speech
        r3 = vad.process_chunk(make_pcm16_chunk(duration=0.1))
        assert not r3.speech_end
        assert vad.is_speaking

    def test_07_vad_state_reset(self) -> None:
        """Reset clears accumulators and speaking state."""
        vad = MockVADProvider(
            threshold=0.5,
            min_speech_duration=0.1,
            canned_probabilities=[0.9],
        )
        vad.process_chunk(make_pcm16_chunk(duration=0.1))
        assert vad.is_speaking

        vad.reset()
        assert not vad.is_speaking
        assert vad._speech_accumulator == 0.0
        assert vad._silence_accumulator == 0.0
        assert vad._timeline_pos == 0.0

    def test_08_empty_audio_chunk_rejected(self) -> None:
        """Passing empty data to VAD raises InvalidAudioDataError."""
        vad = MockVADProvider()
        chunk = AudioChunk(data=b"", sample_rate=16000, duration=0.0)
        with pytest.raises(InvalidAudioDataError):
            vad.process_chunk(chunk)

    def test_09_vad_simulated_computation_error(self) -> None:
        """Simulated provider failure raises VADProcessingError."""
        vad = MockVADProvider(force_error=True)
        chunk = make_pcm16_chunk(duration=0.1)
        with pytest.raises(VADProcessingError):
            vad.process_chunk(chunk)


# =============================================================================
# 2. Speech-to-Text (STT) Tests
# =============================================================================

class TestSTTProvider:
    """Unit tests for STT transcription, input validations, schema verification, and latency."""

    @pytest.mark.asyncio
    async def test_10_valid_transcription(self) -> None:
        """Valid audio buffer produces structured STTResult model."""
        stt = MockSTTProvider(canned_transcriptions=["I would like to learn about AI."])
        chunk = make_pcm16_chunk(duration=1.0)
        result = await stt.transcribe(chunk)

        assert isinstance(result, STTResult)
        assert result.text == "I would like to learn about AI."
        assert result.confidence == 0.95
        assert result.language == "en"
        assert result.audio_duration == 1.0
        assert result.processing_time is not None
        assert result.real_time_factor is not None
        assert result.is_final is True

    @pytest.mark.asyncio
    async def test_11_empty_audio_buffer_rejected(self) -> None:
        """Empty audio buffer raises InvalidAudioDataError."""
        stt = MockSTTProvider()
        with pytest.raises(InvalidAudioDataError):
            await stt.transcribe(b"")

    @pytest.mark.asyncio
    async def test_12_insufficient_audio_buffer_rejected(self) -> None:
        """Audio buffer with less than 2 bytes raises InvalidAudioDataError."""
        stt = MockSTTProvider()
        with pytest.raises(InvalidAudioDataError):
            await stt.transcribe(b"\x00")

    @pytest.mark.asyncio
    async def test_13_provider_failure_propagation(self) -> None:
        """STT provider internal exception is wrapped in STTTranscriptionError."""
        stt = MockSTTProvider(force_error=True)
        chunk = make_pcm16_chunk(duration=0.5)
        with pytest.raises(STTTranscriptionError):
            await stt.transcribe(chunk)

    @pytest.mark.asyncio
    async def test_14_transcribe_speech_segment_instance(self) -> None:
        """STT transcribe cleanly accepts SpeechSegment instances."""
        stt = MockSTTProvider(default_text="Testing SpeechSegment transcription")
        chunk = make_pcm16_chunk(duration=0.5)
        segment = SpeechSegment(
            audio_data=chunk.data,
            sample_rate=16000,
            channels=1,
            start_time=1.0,
            end_time=1.5,
            duration=0.5,
            chunks_count=1,
        )
        result = await stt.transcribe(segment)
        assert result.text == "Testing SpeechSegment transcription"
        assert result.audio_duration == 0.5

    @pytest.mark.asyncio
    async def test_15_stt_result_string_conversion(self) -> None:
        """STTResult str() returns transcribed text for backward compatibility."""
        res = STTResult(text="Hello world")
        assert str(res) == "Hello world"


# =============================================================================
# 3. Text-to-Speech (TTS) Tests
# =============================================================================

class TestTTSProvider:
    """Unit tests for TTS speech synthesis, WAV byte containers, cancellation, and streaming."""

    @pytest.mark.asyncio
    async def test_16_valid_synthesis(self) -> None:
        """Valid text input produces valid playable WAV audio payload."""
        tts = MockTTSProvider(sample_rate=24000)
        result = await tts.synthesize("Welcome to our service.")

        assert isinstance(result, TTSResult)
        assert len(result.audio_data) > 44  # WAV header + PCM frames
        assert result.audio_data[:4] == b"RIFF"  # Valid RIFF header
        assert result.sample_rate == 24000
        assert result.format == "wav"
        assert result.duration is not None and result.duration > 0.0
        assert result.processing_time is not None
        assert result.real_time_factor is not None

    @pytest.mark.asyncio
    async def test_17_empty_text_rejected(self) -> None:
        """Synthesizing empty or whitespace text raises InvalidAudioDataError."""
        tts = MockTTSProvider()
        with pytest.raises(InvalidAudioDataError):
            await tts.synthesize("")

        with pytest.raises(InvalidAudioDataError):
            await tts.synthesize("   \n\t  ")

    @pytest.mark.asyncio
    async def test_18_tts_failure_propagation(self) -> None:
        """TTS provider internal exception is wrapped in TTSSynthesisError."""
        tts = MockTTSProvider(force_error=True)
        with pytest.raises(TTSSynthesisError):
            await tts.synthesize("This will fail.")

    @pytest.mark.asyncio
    async def test_19_cancellation_readiness(self) -> None:
        """Barge-in cancellation signal aborts synthesis."""
        tts = MockTTSProvider()
        tts.cancel_current_synthesis()
        assert tts._is_cancelled is True

        with pytest.raises(TTSSynthesisError):
            await tts.synthesize("Caller interrupted agent speech.")

        # Flag is reset after being consumed or explicitly
        tts.reset_cancellation()
        assert tts._is_cancelled is False
        # Next call succeeds
        res = await tts.synthesize("Fresh turn after interruption.")
        assert res.audio_data[:4] == b"RIFF"

    @pytest.mark.asyncio
    async def test_20_streaming_synthesis(self) -> None:
        """Streaming text synthesis yields audio byte chunks."""
        tts = MockTTSProvider()

        async def text_generator() -> AsyncIterator[str]:
            yield "First sentence."
            yield "Second sentence."

        chunks = []
        async for chunk in tts.stream_synthesize(text_generator()):
            chunks.append(chunk)

        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk[:4] == b"RIFF"


# =============================================================================
# 4. Audio Pipeline (Framing & Segmentation) Tests
# =============================================================================

class TestAudioPipeline:
    """Unit tests for AudioChunk -> VAD -> SpeechSegment -> STT pipeline."""

    @pytest.mark.asyncio
    async def test_21_pipeline_speech_segment_and_transcription(self) -> None:
        """Contiguous speech frames trigger speech_start, accumulation, and STT on speech_end."""
        # 2 frames speech (0.2s), 2 frames silence (0.2s)
        vad = MockVADProvider(
            min_speech_duration=0.2,
            min_silence_duration=0.2,
            canned_probabilities=[0.9, 0.9, 0.1, 0.1],
        )
        stt = MockSTTProvider(canned_transcriptions=["End-to-end pipeline test"])
        pipeline = AudioPipeline(vad_provider=vad, stt_provider=stt)

        # Frame 1: speech onset building
        v1, s1 = await pipeline.process_chunk(make_pcm16_chunk(duration=0.1))
        assert not v1.speech_start
        assert s1 is None

        # Frame 2: speech onset reached -> speech_start=True
        v2, s2 = await pipeline.process_chunk(make_pcm16_chunk(duration=0.1))
        assert v2.speech_start
        assert s2 is None
        assert pipeline._speech_active

        # Frame 3: silence onset building
        v3, s3 = await pipeline.process_chunk(make_silence_chunk(duration=0.1))
        assert not v3.speech_end
        assert s3 is None

        # Frame 4: silence duration reached -> speech_end=True -> triggers STT
        v4, s4 = await pipeline.process_chunk(make_silence_chunk(duration=0.1))
        assert v4.speech_end
        assert s4 is not None
        assert s4.text == "End-to-end pipeline test"
        assert not pipeline._speech_active

    @pytest.mark.asyncio
    async def test_22_pipeline_silence_does_not_trigger_stt(self) -> None:
        """Pure silence frames never trigger STT transcription."""
        vad = MockVADProvider(canned_probabilities=[0.05, 0.05, 0.05])
        stt = MockSTTProvider()
        pipeline = AudioPipeline(vad_provider=vad, stt_provider=stt)

        for _ in range(3):
            v, s = await pipeline.process_chunk(make_silence_chunk(duration=0.1))
            assert not v.is_speech
            assert s is None

        assert len(stt.transcribed_calls) == 0

    @pytest.mark.asyncio
    async def test_23_pipeline_flush_emits_active_speech(self) -> None:
        """Flushing an ongoing speech segment triggers STT on session close."""
        vad = MockVADProvider(min_speech_duration=0.1, canned_probabilities=[0.9])
        stt = MockSTTProvider(canned_transcriptions=["Flushed utterance"])
        pipeline = AudioPipeline(vad_provider=vad, stt_provider=stt)

        await pipeline.process_chunk(make_pcm16_chunk(duration=0.1))
        assert pipeline._speech_active

        flushed_result = await pipeline.flush()
        assert flushed_result is not None
        assert flushed_result.text == "Flushed utterance"
        assert not pipeline._speech_active

    @pytest.mark.asyncio
    async def test_24_pipeline_error_wrapping(self) -> None:
        """Failures in VAD or STT inside the pipeline are wrapped in AudioPipelineError."""
        vad = MockVADProvider(force_error=True)
        stt = MockSTTProvider()
        pipeline = AudioPipeline(vad_provider=vad, stt_provider=stt)

        with pytest.raises(AudioPipelineError):
            await pipeline.process_chunk(make_pcm16_chunk(duration=0.1))

    def test_25_pipeline_reset(self) -> None:
        """Pipeline reset clears buffers and internal VAD state."""
        vad = MockVADProvider()
        stt = MockSTTProvider()
        pipeline = AudioPipeline(vad_provider=vad, stt_provider=stt)
        pipeline._speech_active = True
        pipeline._speech_chunks = [make_pcm16_chunk(duration=0.1)]

        pipeline.reset()
        assert not pipeline._speech_active
        assert len(pipeline._speech_chunks) == 0


# =============================================================================
# 5. Workbench Service & Bootstrap Integration Tests
# =============================================================================

class TestWorkbenchAudioIntegration:
    """Verify Streamlit WorkbenchService audio helper methods and bootstrap wiring."""

    def test_26_bootstrap_initializes_audio_providers(self) -> None:
        """Bootstrap creates WorkbenchService equipped with mock audio providers."""
        service = bootstrap_workbench(force_mock=True)

        assert service.vad_provider is not None
        assert service.stt_provider is not None
        assert service.tts_provider is not None
        assert service.vad_mode == "MOCK"
        assert service.stt_mode == "MOCK"
        assert service.tts_mode == "MOCK"

    def test_27_workbench_test_vad_method(self) -> None:
        """WorkbenchService.test_vad evaluates audio chunk properly."""
        service = bootstrap_workbench(force_mock=True)
        chunk = make_pcm16_chunk(duration=0.1)
        res = service.test_vad(chunk.data)
        assert isinstance(res, VADResult)

    def test_28_workbench_test_stt_method(self) -> None:
        """WorkbenchService.test_stt returns transcription."""
        service = bootstrap_workbench(force_mock=True)
        chunk = make_pcm16_chunk(duration=0.5)
        res = service.test_stt(chunk.data)
        assert isinstance(res, STTResult)
        assert len(res.text) > 0

    def test_29_workbench_test_tts_method(self) -> None:
        """WorkbenchService.test_tts returns valid WAV bytes."""
        service = bootstrap_workbench(force_mock=True)
        res = service.test_tts("Testing audio synthesis from workbench.")
        assert isinstance(res, TTSResult)
        assert res.audio_data[:4] == b"RIFF"

    def test_30_debug_payload_includes_audio_diagnostics(self) -> None:
        """Debug payload exposes VAD, STT, and TTS modes."""
        service = bootstrap_workbench(force_mock=True)
        state = service.create_inbound_session("test-audio-call", "+15550001111")
        debug = service.get_debug_payload(state=state)

        assert "vad_mode" in debug["runtime"]
        assert "stt_mode" in debug["runtime"]
        assert "tts_mode" in debug["runtime"]
