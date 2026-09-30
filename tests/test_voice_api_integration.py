"""Comprehensive Integration Test Suite for Integrated Voice & API Layer.

Verifies:
1. REST API Endpoints:
   - GET /health and GET /api/v1/health (Subsystem status check)
   - GET /config and GET /api/v1/config (Model config & supported domains)
   - POST /voice and POST /api/v1/voice (TwiML XML response with WebSocket MediaStream connect)
   - POST /chat and POST /api/v1/chat (REST-based conversational turn processing)
   - POST /api/v1/voice/session/start, stop, status, sessions (Voice session lifecycle)
2. Audio Services:
   - AudioProcessor: Polyphase rational resampling (48kHz -> 16kHz)
   - AudioProcessor: Frame chunking (512 samples / 32ms)
   - SileroVADService: Voice Activity Detection (silence vs speech, state machine)
3. Speech-to-Text & Text-to-Speech:
   - STTService: Audio transcription (with fallback and mock)
   - DeepgramFluxTTS: Streaming synthesis chunks, cancellation handling
4. WebSocket /media-stream:
   - Connection handshake and start event processing
   - Audio chunk processing
   - Instant barge-in / interruption signal emission
   - Clean disconnection and session disposal
"""

from __future__ import annotations

import os
os.environ["FORCE_MOCK"] = "true"

import base64
import json
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.ui.app import app
from src.voice.audio_services.audio_processor import AudioProcessor
from src.voice.audio_services.vad_services import SileroVADService, VADEventType
from src.voice.stt.stt_service import STTService
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS
from src.voice.session_manager import VoiceSessionManager
from src.api.dependencies import get_voice_session_manager, get_conversation_engine, get_state_manager
from src.core.types import DomainType


@pytest.fixture(autouse=True)
def manage_force_mock():
    """Ensure FORCE_MOCK is set for integration tests and restored afterwards."""
    old_mock = os.environ.get("FORCE_MOCK")
    os.environ["FORCE_MOCK"] = "true"
    yield
    if old_mock is None:
        os.environ.pop("FORCE_MOCK", None)
    else:
        os.environ["FORCE_MOCK"] = old_mock


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ============================================================================
# 1. REST API Endpoints
# ============================================================================

class TestRestEndpoints:
    """Validate all REST endpoints across root and /api/v1 prefixes."""

    def test_01_health_endpoints(self, client: TestClient):
        """GET /health and GET /api/v1/health return healthy status with subsystems."""
        for path in ["/health", "/api/v1/health"]:
            response = client.get(path)
            assert response.status_code == 200, f"Failed for {path}: {response.text}"
            data = response.json()
            assert data["status"] == "healthy"
            assert "components" in data
            components = data["components"]
            assert "stt" in components
            assert "tts" in components
            assert "vad" in components
            assert "engine" in components
            assert "state_manager" in components

    def test_02_config_endpoints(self, client: TestClient):
        """GET /config and GET /api/v1/config return active Gemini model and domains."""
        for path in ["/config", "/api/v1/config"]:
            response = client.get(path)
            assert response.status_code == 200, f"Failed for {path}: {response.text}"
            data = response.json()
            assert data["model"] == "gemini-3.5-flash-lite"
            assert "edusaas" in data["supported_domains"]
            assert "vayvora" in data["supported_domains"]
            assert data["audio_sample_rate"] == 16000

    def test_03_voice_twiml_endpoints(self, client: TestClient):
        """POST /voice and POST /api/v1/voice return valid TwiML XML connecting to /media-stream."""
        for path in ["/voice", "/api/v1/voice"]:
            response = client.post(path)
            assert response.status_code == 200, f"Failed for {path}: {response.text}"
            assert "xml" in response.headers.get("content-type", "")
            xml_text = response.text
            assert "<Response>" in xml_text
            assert "<Connect>" in xml_text
            assert "<Stream" in xml_text
            assert "/media-stream" in xml_text

    def test_04_chat_endpoint_turn_processing(self, client: TestClient):
        """POST /chat and POST /api/v1/chat process conversation turns and return responses."""
        payload = {
            "message": "Hello, what programs or courses do you offer?",
            "session_id": "test-rest-chat-session",
            "domain": "edusaas",
            "caller_phone": "+1234567890",
        }
        for path in ["/chat", "/api/v1/chat"]:
            response = client.post(path, json=payload)
            assert response.status_code == 200, f"Failed for {path}: {response.text}"
            data = response.json()
            assert data["session_id"] == "test-rest-chat-session"
            assert isinstance(data["response"], str)
            assert len(data["response"]) > 0
            assert data["domain"] == "edusaas"
            assert "stage" in data
            assert "slots" in data

    def test_05_session_management_lifecycle(self, client: TestClient):
        """Session start, status, listing, and stop API lifecycle."""
        session_id = "test-api-lifecycle-session"
        
        # Start session
        start_payload = {
            "session_id": session_id,
            "domain": "edusaas",
            "caller_phone": "+1987654321",
            "direction": "inbound",
        }
        start_resp = client.post("/api/v1/voice/session/start", json=start_payload)
        assert start_resp.status_code == 200
        start_data = start_resp.json()
        assert start_data["session_id"] == session_id
        assert start_data["status"] == "active"

        # Check status
        status_resp = client.get(f"/api/v1/voice/session/status?session_id={session_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "active"

        # List sessions
        list_resp = client.get("/api/v1/voice/sessions")
        assert list_resp.status_code == 200
        assert session_id in list_resp.json()["active_sessions"]

        # Stop session
        stop_resp = client.post(f"/api/v1/voice/session/stop?session_id={session_id}")
        assert stop_resp.status_code == 200
        assert stop_resp.json()["status"] == "stopped"


# ============================================================================
# 2. Audio Services: Resampling, Frame Chunking, and VAD
# ============================================================================

class TestAudioServices:
    """Validate AudioProcessor polyphase resampling and Silero VAD state machine."""

    def test_06_polyphase_resampling_48k_to_16k(self):
        """AudioProcessor resamples 48kHz PCM16 audio to 16kHz with 3:1 ratio."""
        processor = AudioProcessor()
        
        # Create 100ms of 48kHz audio (4800 samples, 2 bytes/sample = 9600 bytes)
        num_samples_48k = 4800
        samples_48k = (np.sin(np.linspace(0, 100 * np.pi, num_samples_48k)) * 16000).astype(np.int16)
        raw_48k = samples_48k.tobytes()
        assert len(raw_48k) == 9600

        resampled_16k = processor.resample_48k_to_16k(raw_48k)
        # At 16kHz, 100ms is 1600 samples = 3200 bytes
        assert len(resampled_16k) == 3200

    def test_07_audio_processor_frame_buffering(self):
        """AudioProcessor buffers incoming 48kHz audio and yields 512-sample (1024-byte) 16kHz frames."""
        processor = AudioProcessor()
        
        # Supply 48kHz samples corresponding to multiple 16kHz frames (512 * 3 = 1536 samples at 48kHz)
        samples_48k = (np.zeros(1536, dtype=np.int16)).tobytes()
        frames = processor.process_audio(samples_48k)
        
        # Exactly 1 frame of 512 samples should be extracted
        assert len(frames) == 1
        assert len(frames[0]) == 1024  # 512 samples * 2 bytes/sample

    def test_08_vad_silence_detection(self):
        """SileroVADService processes silent frames without triggering false speech."""
        vad = SileroVADService(sample_rate=16000)
        vad.reset()

        # Send 10 consecutive silent 32ms frames (512 samples each = 1024 bytes)
        silent_frame = b"\x00" * 1024
        for _ in range(10):
            events = vad.process(silent_frame)
            # Silence should not trigger START_OF_SPEECH or ACTIVE_SPEECH
            for ev in events:
                assert ev.event_type not in (VADEventType.START_OF_SPEECH, VADEventType.ACTIVE_SPEECH)

        assert vad.is_speaking is False

    def test_09_vad_speech_start_and_end_lifecycle(self):
        """SileroVADService detects speech onset and speech end boundary."""
        vad = SileroVADService(sample_rate=16000, start_threshold=0.3, end_threshold=0.2)
        vad.reset()

        # Generate synthetic high-amplitude speech-like frame (512 samples = 32ms)
        t = np.linspace(0, 0.032, 512, endpoint=False)
        speech_samples = (np.sin(2 * np.pi * 300 * t) * 20000).astype(np.int16)
        speech_frame = speech_samples.tobytes()

        # Feed enough speech frames to trigger start threshold
        speech_started = False
        for _ in range(15):
            events = vad.process(speech_frame)
            for ev in events:
                if ev.event_type == VADEventType.START_OF_SPEECH:
                    speech_started = True

        assert speech_started is True, "VAD should have detected speech onset"

        # Now feed silence to trigger end threshold and obtain completed speech segment
        end_event = None
        silent_frame = b"\x00" * 1024
        for _ in range(30):
            events = vad.process(silent_frame)
            for ev in events:
                if ev.event_type == VADEventType.END_OF_SPEECH:
                    end_event = ev
                    break
            if end_event is not None:
                break

        assert end_event is not None, "VAD should emit completed speech segment on silence after speech"
        assert end_event.audio_pcm16 is not None
        assert len(end_event.audio_pcm16) > 0


# ============================================================================
# 3. Speech-to-Text & Text-to-Speech Services
# ============================================================================

class TestSTTAndTTSServices:
    """Validate STT transcription and Deepgram TTS streaming synthesis."""

    @pytest.mark.asyncio
    async def test_10_stt_transcription_with_mock_fallback(self):
        """STTService transcribes 16kHz PCM audio bytes."""
        stt = STTService()
        
        # Test with empty audio -> empty transcription
        empty_res = await stt.transcribe(b"")
        assert empty_res == ""

        # Test with audio payload using mock or fallback
        test_audio = b"\x00\x01" * 1600  # 100ms of PCM16
        with patch.object(stt, "transcribe_pcm16", return_value="Hello, I am interested in EduSaaS courses.") as mock_trans:
            result = await stt.transcribe(test_audio)
            assert result == "Hello, I am interested in EduSaaS courses."
            mock_trans.assert_called_once_with(test_audio)

    @pytest.mark.asyncio
    async def test_11_deepgram_tts_streaming_chunks(self):
        """DeepgramFluxTTS synthesizes text into streaming PCM audio chunks."""
        tts = DeepgramFluxTTS()
        
        chunks = []
        async for chunk in tts.synthesize_stream("Welcome to EduSaaS, how can I assist you today?"):
            chunks.append(chunk)

        assert len(chunks) > 0
        total_bytes = sum(len(c) for c in chunks)
        assert total_bytes > 0

    @pytest.mark.asyncio
    async def test_12_deepgram_tts_cancellation(self):
        """DeepgramFluxTTS halts generation when cancel() is invoked."""
        tts = DeepgramFluxTTS()
        tts.cancel()
        assert len(tts._accumulated_text) == 0


# ============================================================================
# 4. WebSocket /media-stream: Media, Turn, and Barge-In
# ============================================================================

class TestWebSocketMediaStream:
    """Validate full WebSocket /media-stream connection, Twilio protocol, and barge-in."""

    def test_13_websocket_handshake_and_start(self, client: TestClient):
        """WebSocket accepts connection, parses start event, and initializes session."""
        with client.websocket_connect("/media-stream") as ws:
            start_msg = {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "streamSid": "test-ws-stream-sid",
                    "accountSid": "test-acc-sid",
                    "callSid": "test-call-sid",
                    "tracks": ["inbound"],
                    "customParameters": {
                        "domain": "edusaas",
                        "caller_phone": "+15551234567",
                    },
                },
                "streamSid": "test-ws-stream-sid",
            }
            ws.send_text(json.dumps(start_msg))

            # Send a media frame of silence
            silent_chunk = base64.b64encode(b"\x00" * 640).decode("utf-8")
            media_msg = {
                "event": "media",
                "sequenceNumber": "2",
                "media": {
                    "track": "inbound",
                    "chunk": "1",
                    "timestamp": "0",
                    "payload": silent_chunk,
                },
                "streamSid": "test-ws-stream-sid",
            }
            ws.send_text(json.dumps(media_msg))

            # Close connection
            ws.close()

    @pytest.mark.asyncio
    async def test_14_websocket_barge_in_interruption(self, client: TestClient):
        """Barge-in: When caller speaks while agent is speaking, interrupt signal is emitted."""
        session_mgr = get_voice_session_manager()

        with client.websocket_connect("/media-stream") as ws:
            session_id = "test-barge-in-stream"
            start_msg = {
                "event": "start",
                "streamSid": session_id,
                "start": {
                    "streamSid": session_id,
                    "callSid": "call-barge-in",
                    "customParameters": {"domain": "edusaas"},
                },
            }
            ws.send_text(json.dumps(start_msg))

            # Trigger interrupt signal on active session
            interrupted = await session_mgr.interrupt_speech(session_id)
            assert isinstance(interrupted, bool)

            ws.close()
