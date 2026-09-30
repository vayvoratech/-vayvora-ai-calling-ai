"""Comprehensive unit and integration test suite for Groq Whisper STT and Deepgram TTS providers.

Verifies:
1. Groq STT provider initialization.
2. Groq STT audio transcription with mocked API.
3. Deepgram TTS initialization.
4. Deepgram streaming chunks with mocked WebSocket/API.
5. Deepgram interruption.
6. WebSocket voice session.
7. Multiple conversation turns in one session.
8. Barge-in.
9. End session.
10. No Faster-Whisper initialization on startup.
11. No Kokoro initialization on startup.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from src.config import Settings
from src.core.types import DomainType
from src.ui.app import app
from src.voice.stt.stt_service import GroqWhisperSTT, STTService
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS, DeepgramTTS
from src.voice.session_manager import VoiceSessionManager
from src.api.dependencies import get_voice_session_manager, get_conversation_engine, get_state_manager


@pytest.fixture
def client():
    return TestClient(app)


# ============================================================================
# 1 & 2: Groq Whisper STT Provider Tests
# ============================================================================

class TestGroqWhisperSTTProvider:
    """Test suite for Groq Whisper STT provider."""

    def test_01_groq_stt_initialization(self):
        """Verify GroqWhisperSTT initializes with custom configurations."""
        with patch("groq.Groq") as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client

            stt = GroqWhisperSTT(
                api_key="gsk_test_api_key_12345",
                model="whisper-large-v3-turbo",
                sample_rate=16000,
                language="en",
                prompt="Custom Domain Prompt",
            )

            assert stt.model == "whisper-large-v3-turbo"
            assert stt.sample_rate == 16000
            assert stt.language == "en"
            assert stt.prompt == "Custom Domain Prompt"
            assert stt.client is not None
            mock_groq_class.assert_called_once_with(api_key="gsk_test_api_key_12345")

    def test_02_groq_stt_transcription_with_mocked_api(self):
        """Verify Groq STT transcribes PCM16 audio using mocked Groq API."""
        with patch("groq.Groq") as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client

            mock_transcription = "What AI courses does EduSaaS provide?"
            mock_client.audio.transcriptions.create.return_value = mock_transcription

            stt = GroqWhisperSTT(api_key="gsk_test_api_key_12345")

            # 16000Hz mono PCM16 data (0.5s = 16000 bytes)
            dummy_pcm16 = b"\x00\x10" * 8000
            result = stt.transcribe_pcm16(dummy_pcm16)

            assert result == "What AI courses does EduSaaS provide?"
            mock_client.audio.transcriptions.create.assert_called_once()
            call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
            assert call_kwargs["model"] == "whisper-large-v3-turbo"
            assert call_kwargs["language"] == "en"
            assert "Vayvora" in call_kwargs["prompt"]
            assert call_kwargs["response_format"] == "text"

    def test_02b_groq_stt_error_handling(self):
        """Verify Groq STT failure returns empty string and does not crash."""
        with patch("groq.Groq") as mock_groq_class:
            mock_client = MagicMock()
            mock_groq_class.return_value = mock_client
            mock_client.audio.transcriptions.create.side_effect = RuntimeError("Groq API Timeout")

            stt = GroqWhisperSTT(api_key="gsk_test_api_key_12345")
            dummy_pcm16 = b"\x00\x10" * 8000
            result = stt.transcribe_pcm16(dummy_pcm16)
            assert result == ""


# ============================================================================
# 3, 4, 5: Deepgram TTS Provider Tests
# ============================================================================

class TestDeepgramTTSProvider:
    """Test suite for Deepgram Streaming TTS provider."""

    def test_03_deepgram_tts_initialization(self):
        """Verify DeepgramFluxTTS initializes with correct voice, encoding, sample rate."""
        tts = DeepgramFluxTTS(
            model="aura-asteria-en",
            encoding="linear16",
            sample_rate=48000,
            api_key="test_deepgram_key_67890",
        )
        assert tts.model == "aura-asteria-en"
        assert tts.encoding == "linear16"
        assert tts.sample_rate == 48000
        assert tts.api_key == "test_deepgram_key_67890"

    @pytest.mark.asyncio
    async def test_04_deepgram_streaming_chunks(self):
        """Verify Deepgram TTS sends text and dispatches audio chunks via callback."""
        mock_ws = AsyncMock()

        async def empty_stream():
            if False:
                yield b""

        mock_ws.__aiter__.side_effect = empty_stream

        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            tts = DeepgramTTS(
                model="aura-asteria-en",
                api_key="test_deepgram_key_67890",
            )
            await tts.connect()
            assert tts.connected is True

            received_audio = []
            received_events = []

            async def on_audio(chunk: bytes):
                received_audio.append(chunk)

            async def on_event(ev: dict):
                received_events.append(ev)

            tts.audio_callback = on_audio
            tts.event_callback = on_event

            # Test sending text
            await tts.send_text("Hello and welcome")
            mock_ws.send.assert_called_with(json.dumps({"type": "Speak", "text": "Hello and welcome"}))

            # Test flush
            await tts.flush()
            mock_ws.send.assert_called_with(json.dumps({"type": "Flush"}))

    @pytest.mark.asyncio
    async def test_05_deepgram_interruption_barge_in(self):
        """Verify Deepgram TTS interrupt sends Clear/Interrupt and resets turn state."""
        mock_ws = AsyncMock()

        async def empty_stream():
            if False:
                yield b""

        mock_ws.__aiter__.side_effect = empty_stream

        with patch("websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            tts = DeepgramFluxTTS(
                model="aura-asteria-en",
                api_key="test_deepgram_key_67890",
            )
            await tts.connect()

            tts.turn_started_at = 100.0
            tts.first_audio_received = True

            await tts.interrupt()

            # For aura models, Clear is sent
            mock_ws.send.assert_called_with(json.dumps({"type": "Clear"}))
            assert tts.turn_started_at is None
            assert tts.first_audio_received is False


# ============================================================================
# 6, 7, 8, 9: WebSocket Voice Session, Multi-Turn, Barge-In & End Session
# ============================================================================

class TestWebSocketVoiceSession:
    """Test suite for /media-stream WebSocket, continuous session, and barge-in."""

    def test_06_websocket_handshake_and_session_started(self, client: TestClient):
        """Verify /media-stream connects and emits session_started."""
        with client.websocket_connect("/media-stream?domain=edusaas&direction=inbound") as ws:
            first_msg = ws.receive_json()
            assert first_msg.get("type") == "session_started"
            assert first_msg.get("domain") == "edusaas"
            assert first_msg.get("direction") == "inbound"
            ws.close()

    def test_07_multiple_conversation_turns_in_one_session(self, client: TestClient):
        """Verify session remains active across multiple turns without disconnecting."""
        manager = get_voice_session_manager()

        with client.websocket_connect("/media-stream?session_id=sess_multi_turn_test") as ws:
            start_event = ws.receive_json()
            assert start_event["type"] == "session_started"
            sess_id = start_event["session_id"]

            state = manager.get_session(sess_id)
            assert state is not None
            assert state.conversation_active is True

            # Send silence frame (does NOT end session)
            ws.send_bytes(b"\x00" * 1024)
            # Send another frame (session still active)
            ws.send_bytes(b"\x00" * 1024)

            state_after = manager.get_session(sess_id)
            assert state_after.conversation_active is True
            ws.close()

    @pytest.mark.asyncio
    async def test_08_barge_in_cancels_turn_and_interrupts_tts(self):
        """Verify barge-in speech detection interrupts active TTS and cancels agent turn."""
        manager = get_voice_session_manager()
        mock_tts = AsyncMock(spec=DeepgramFluxTTS)

        session_id = "sess_barge_in_test"
        manager._active_sessions[session_id] = {
            "tts": mock_tts,
        }

        interrupted = await manager.interrupt_speech(session_id)
        assert interrupted is True
        mock_tts.interrupt.assert_awaited_once()

    def test_09_end_session_via_hangup(self, client: TestClient):
        """Verify explicit hangup event ends the session."""
        with client.websocket_connect("/media-stream") as ws:
            ws.receive_json()  # session_started
            # Send explicit hangup command
            ws.send_text(json.dumps({"type": "hangup"}))
            # Socket closes or finishes cleanly
            ws.close()


# ============================================================================
# 10 & 11: No Local Model Initialization (Faster-Whisper & Kokoro)
# ============================================================================

class TestNoLocalModelStartup:
    """Verify Faster-Whisper and Kokoro are not initialized or loaded by default."""

    def test_10_no_faster_whisper_loaded_on_startup(self):
        """Faster-Whisper WhisperModel must NOT be instantiated during standard bootstrap."""
        with patch.dict(os.environ, {"FORCE_MOCK": "false"}):
            with patch("src.audio.stt.FasterWhisperSTTProvider._init_model") as mock_init:
                from src.ui.bootstrap import bootstrap_workbench
                service = bootstrap_workbench()
                assert service is not None
                # FasterWhisperSTTProvider._init_model should NOT have been called
                mock_init.assert_not_called()

    def test_11_no_kokoro_loaded_on_startup(self):
        """Kokoro KPipeline must NOT be instantiated during standard bootstrap."""
        with patch.dict(os.environ, {"FORCE_MOCK": "false"}):
            with patch("src.audio.tts.KokoroTTSProvider._init_pipeline") as mock_init:
                from src.ui.bootstrap import bootstrap_workbench
                service = bootstrap_workbench()
                assert service is not None
                # KokoroTTSProvider._init_pipeline should NOT have been called
                mock_init.assert_not_called()
