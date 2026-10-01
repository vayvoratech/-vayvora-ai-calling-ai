"""Integration test for the dedicated Voice Testing UI page.

Verifies:
1. GET /voice-test returns 200 and loads voice_test.html.
2. GET /voice-testing returns 200 and loads voice_test.html.
3. Both endpoints render essential controls: Start Session, End Session,
   visualizer canvas, status badges, transcript history, and WebSocket hook.
4. GET / continues to serve the original workbench UI (index.html) intact.
"""

import pytest
from starlette.testclient import TestClient

from src.ui.app import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


class TestVoiceTestPage:
    """Test suite for voice testing UI routes and content."""

    def test_get_voice_test_route_returns_200_and_expected_elements(self, client: TestClient):
        """GET /voice-test should return 200 with the dedicated voice testing console."""
        response = client.get("/voice-test")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

        html = response.text
        # Verify title and branding
        assert "Vayvora AI" in html
        assert "VOICE TEST" in html

        # Verify essential interactive controls
        assert 'id="btn-start-session"' in html
        assert 'id="btn-end-session"' in html
        assert 'id="sel-domain"' in html
        assert 'id="sel-direction"' in html
        assert 'id="input-phone"' in html

        # Verify visualizer & real-time monitoring elements
        assert 'id="audio-visualizer"' in html
        assert 'id="dot-ws"' in html
        assert 'id="dot-session"' in html
        assert 'id="pill-mic"' in html
        assert 'id="pill-speaker"' in html
        assert 'id="text-state"' in html
        assert 'id="chat-container"' in html
        assert 'id="debug-console"' in html

        # Verify WebSocket and audio pipeline scripts
        assert "/media-stream" in html
        assert "navigator.mediaDevices.getUserMedia" in html
        assert "AudioContext" in html
        assert "interrupt" in html
        assert "tts_interrupted" in html

    def test_get_voice_testing_alias_returns_200(self, client: TestClient):
        """GET /voice-testing alias should return 200 with identical testing interface."""
        response = client.get("/voice-testing")
        assert response.status_code == 200
        assert 'id="btn-start-session"' in response.text
        assert 'id="btn-end-session"' in response.text
        assert 'id="audio-visualizer"' in response.text

    def test_original_workbench_route_preserved(self, client: TestClient):
        """GET / should continue serving the original testing workbench without changes."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        html = response.text
        # Original workbench uses "Unified AI Voice Agent" or "Workbench Console"
        assert "Unified AI Voice Agent" in html
        assert "Chat History" in html or "workbench" in html.lower()
