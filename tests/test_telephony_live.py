"""Optional Live Integration Tests for External Telephony Transport.

Executes live media streaming over WebSocket telephony transport when an active
telephony gateway endpoint is configured in the environment (e.g. TELEPHONY_ENDPOINT).
Gracefully skips automatically when no live external transport is configured.
"""

import os
import pytest

from src.config import get_settings
from src.core.types import CallDirection
from src.telephony.transport import GenericWebSocketTelephonyTransport

settings = get_settings()
HAS_LIVE_TELEPHONY_ENDPOINT = bool(
    os.getenv("TELEPHONY_ENDPOINT")
    or (hasattr(settings, "telephony_endpoint") and settings.telephony_endpoint)
)


@pytest.mark.skipif(
    not HAS_LIVE_TELEPHONY_ENDPOINT,
    reason="No live external telephony transport endpoint configured (TELEPHONY_ENDPOINT).",
)
class TestLiveTelephonyTransport:
    """Live media gateway connectivity tests (executed only in environments with active media gateway)."""

    @pytest.mark.asyncio
    async def test_01_live_websocket_connection_handshake(self) -> None:
        """Connects to live telephony media gateway and verifies transport handshake."""
        endpoint = os.getenv("TELEPHONY_ENDPOINT") or settings.telephony_endpoint
        transport = GenericWebSocketTelephonyTransport(
            endpoint_url=endpoint,
            call_id="live-call-01",
            session_id="live-session-01",
            direction=CallDirection.INBOUND,
        )
        await transport.connect()
        assert transport.is_connected is True
        await transport.disconnect()
        assert transport.is_connected is False
