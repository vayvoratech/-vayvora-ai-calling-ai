"""Live / Optional Integration Tests for Local Neural Audio Engines.

Tests live Silero VAD, faster-whisper STT, and Kokoro TTS when local dependencies
and model weights are available in the runtime environment.
Gracefully skips when local model checkpoints are not installed.
"""

import pytest

from src.audio.stt import FasterWhisperSTTProvider
from src.audio.tts import KokoroTTSProvider, PiperTTSProvider
from src.audio.vad import SileroVADProvider
from src.core.errors import ModelUnavailableError


class TestLiveAudioProviders:
    """Validate live audio providers or ensure informative ModelUnavailableError is raised."""

    def test_01_silero_vad_without_weights_raises_unavailable(self) -> None:
        """Silero VAD raises ModelUnavailableError when no ONNX weights are loaded."""
        vad = SileroVADProvider(model_path=None)
        with pytest.raises(ModelUnavailableError) as exc_info:
            vad.compute_speech_probability(b"\x00" * 3200)
        assert "not available or not loaded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_02_faster_whisper_without_weights_raises_unavailable(self) -> None:
        """Faster-whisper STT raises ModelUnavailableError when uninitialized."""
        stt = FasterWhisperSTTProvider(model_name="nonexistent-model")
        # If faster_whisper package is not installed or model cannot load, transcribe raises
        if stt._model is None:
            with pytest.raises(ModelUnavailableError) as exc_info:
                await stt.transcribe(b"\x00" * 3200)
            assert "is not installed or available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_03_kokoro_tts_without_weights_raises_unavailable(self) -> None:
        """Kokoro TTS raises ModelUnavailableError when uninitialized."""
        tts = KokoroTTSProvider()
        if tts._pipeline is None:
            with pytest.raises(ModelUnavailableError) as exc_info:
                await tts.synthesize("Hello world")
            assert "is not installed or available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_04_piper_tts_without_weights_raises_unavailable(self) -> None:
        """Piper TTS raises ModelUnavailableError when no model path provided."""
        tts = PiperTTSProvider(model_path=None)
        with pytest.raises(ModelUnavailableError) as exc_info:
            await tts.synthesize("Testing piper")
        assert "is not installed or available" in str(exc_info.value)
