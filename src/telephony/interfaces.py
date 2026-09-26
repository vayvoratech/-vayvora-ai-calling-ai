"""Abstract base contracts for external telephony transports and session persistence.

Decouples the core voice application from carrier signaling, SIP/WebRTC transports,
and specific database backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.types import CallDirection
from src.telephony.models import CallSummary, TelephonyEvent


class TelephonyTransport(ABC):
    """Abstract interface mediating media and signaling with external telephony infrastructure."""

    @property
    @abstractmethod
    def call_id(self) -> str:
        """Unique external call identifier."""
        pass

    @property
    @abstractmethod
    def session_id(self) -> str:
        """Application voice session identifier."""
        pass

    @property
    @abstractmethod
    def direction(self) -> CallDirection:
        """Inbound or outbound call direction."""
        pass

    @property
    @abstractmethod
    def caller_number(self) -> str:
        """Phone number of the calling party."""
        pass

    @property
    @abstractmethod
    def callee_number(self) -> str:
        """Phone number of the dialed party."""
        pass

    @property
    @abstractmethod
    def connected_at(self) -> Optional[float]:
        """Timestamp when call transport connected."""
        pass

    @property
    @abstractmethod
    def disconnected_at(self) -> Optional[float]:
        """Timestamp when call transport disconnected."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if the transport media stream is active."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Establish media connection with the external carrier or signaling gateway."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Terminate media connection with the external carrier."""
        pass

    @abstractmethod
    async def receive_audio(self) -> Optional[bytes]:
        """Read the next available audio frame from the telephony media stream."""
        pass

    @abstractmethod
    async def send_audio(self, audio_data: bytes) -> None:
        """Transmit synthesized audio frame back through the telephony media stream."""
        pass

    @abstractmethod
    async def send_event(self, event: TelephonyEvent) -> None:
        """Transmit signaling event or DTMF notification to the telephony platform."""
        pass


class CallSessionRepository(ABC):
    """Abstract persistence boundary for storing final call summaries and metadata."""

    @abstractmethod
    async def persist_summary(self, summary: CallSummary) -> None:
        """Persist structured call summary record."""
        pass

    @abstractmethod
    async def get_summary(self, call_id: str) -> Optional[CallSummary]:
        """Retrieve stored call summary by call ID."""
        pass

    @abstractmethod
    async def list_summaries(self, limit: int = 100) -> List[CallSummary]:
        """List historical call session summaries."""
        pass
