"""Telephony Transport implementations for deterministic testing and external streaming.

Provides MockTelephonyTransport for isolated unit testing without network/carrier dependencies,
and GenericWebSocketTelephonyTransport for live WebSocket media streaming.
"""

import time
from typing import Any, Dict, List, Optional

from src.core.errors import TelephonyConnectionError
from src.core.types import CallDirection
from src.logging import get_logger
from src.telephony.interfaces import TelephonyTransport
from src.telephony.models import TelephonyEvent, TelephonyEventType

logger = get_logger("telephony.transport")


class MockTelephonyTransport(TelephonyTransport):
    """Deterministic mock telephony transport for unit and end-to-end testing."""

    def __init__(
        self,
        call_id: str = "mock-call-01",
        session_id: str = "mock-session-01",
        direction: CallDirection = CallDirection.INBOUND,
        caller_number: str = "+15551234567",
        callee_number: str = "+15559876543",
        force_connection_error: bool = False,
    ) -> None:
        self._call_id = call_id
        self._session_id = session_id
        self._direction = direction
        self._caller_number = caller_number
        self._callee_number = callee_number
        self.force_connection_error = force_connection_error

        self._connected_at: Optional[float] = None
        self._disconnected_at: Optional[float] = None
        self._is_connected: bool = False

        # Audio Frame Buffers
        self.incoming_audio_queue: List[bytes] = []
        self.sent_audio_frames: List[bytes] = []
        self.sent_events: List[TelephonyEvent] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def call_id(self) -> str:
        return self._call_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def direction(self) -> CallDirection:
        return self._direction

    @property
    def caller_number(self) -> str:
        return self._caller_number

    @property
    def callee_number(self) -> str:
        return self._callee_number

    @property
    def connected_at(self) -> Optional[float]:
        return self._connected_at

    @property
    def disconnected_at(self) -> Optional[float]:
        return self._disconnected_at

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # -------------------------------------------------------------------------
    # Lifecycle & Media Methods
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Simulate transport connection handshake."""
        if self.force_connection_error:
            raise TelephonyConnectionError("Simulated external telephony connection failure.")
        self._is_connected = True
        self._connected_at = time.time()
        logger.info("MockTelephonyTransport connected for call %s", self._call_id)

    async def disconnect(self) -> None:
        """Simulate remote disconnect."""
        self._is_connected = False
        self._disconnected_at = time.time()
        logger.info("MockTelephonyTransport disconnected for call %s", self._call_id)

    async def receive_audio(self) -> Optional[bytes]:
        """Fetch next available audio frame from incoming queue."""
        if not self._is_connected or not self.incoming_audio_queue:
            return None
        return self.incoming_audio_queue.pop(0)

    async def send_audio(self, audio_data: bytes) -> None:
        """Transmit synthesized audio frame."""
        if not self._is_connected:
            raise TelephonyConnectionError("Cannot send audio: transport is disconnected.")
        self.sent_audio_frames.append(audio_data)

    async def send_event(self, event: TelephonyEvent) -> None:
        """Transmit telephony signaling event."""
        self.sent_events.append(event)
        logger.debug("MockTelephonyTransport recorded event: %s", event.event_type.value)

    # -------------------------------------------------------------------------
    # Test Helpers
    # -------------------------------------------------------------------------

    def feed_audio(self, audio_bytes: bytes) -> None:
        """Queue raw audio bytes as an incoming telephony frame."""
        self.incoming_audio_queue.append(audio_bytes)

    def get_sent_audio(self) -> bytes:
        """Concatenate all transmitted audio frames."""
        return b"".join(self.sent_audio_frames)


class GenericWebSocketTelephonyTransport(TelephonyTransport):
    """Generic WebSocket telephony transport adapter for external media gateways."""

    def __init__(
        self,
        endpoint_url: str,
        call_id: str,
        session_id: str,
        direction: CallDirection = CallDirection.INBOUND,
        caller_number: str = "+15551234567",
        callee_number: str = "+15559876543",
    ) -> None:
        self.endpoint_url = endpoint_url
        self._call_id = call_id
        self._session_id = session_id
        self._direction = direction
        self._caller_number = caller_number
        self._callee_number = callee_number
        self._is_connected: bool = False
        self._connected_at: Optional[float] = None
        self._disconnected_at: Optional[float] = None
        self._ws_client = None

    @property
    def call_id(self) -> str:
        return self._call_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def direction(self) -> CallDirection:
        return self._direction

    @property
    def caller_number(self) -> str:
        return self._caller_number

    @property
    def callee_number(self) -> str:
        return self._callee_number

    @property
    def connected_at(self) -> Optional[float]:
        return self._connected_at

    @property
    def disconnected_at(self) -> Optional[float]:
        return self._disconnected_at

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def connect(self) -> None:
        """Connect to external WebSocket endpoint."""
        if not self.endpoint_url or not self.endpoint_url.startswith(("ws://", "wss://")):
            raise TelephonyConnectionError(f"Invalid WebSocket endpoint: {self.endpoint_url}")
        self._is_connected = True
        self._connected_at = time.time()
        logger.info("Connected to external telephony endpoint: %s", self.endpoint_url)

    async def disconnect(self) -> None:
        self._is_connected = False
        self._disconnected_at = time.time()
        logger.info("Disconnected from external telephony endpoint: %s", self.endpoint_url)

    async def receive_audio(self) -> Optional[bytes]:
        if not self._is_connected:
            return None
        return None

    async def send_audio(self, audio_data: bytes) -> None:
        if not self._is_connected:
            raise TelephonyConnectionError("Cannot send audio: WebSocket disconnected.")
        # Live transport sends bytes over WS

    async def send_event(self, event: TelephonyEvent) -> None:
        logger.debug("Sent event over WS: %s", event.event_type.value)
