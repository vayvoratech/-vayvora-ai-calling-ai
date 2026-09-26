"""External Telephony Transport Integration and Session Persistence package.

Provides telephony transport abstractions, audio framing and format converters,
application-side adapters, and structured call summary persistence.
"""

from src.telephony.adapter import TelephonyVoiceAdapter
from src.telephony.audio import TelephonyAudioConverter
from src.telephony.interfaces import CallSessionRepository, TelephonyTransport
from src.telephony.models import (
    CallObservabilityDiagnostics,
    CallSummary,
    TelephonyEvent,
    TelephonyEventType,
)
from src.telephony.repository import MockCallSessionRepository
from src.telephony.transport import (
    GenericWebSocketTelephonyTransport,
    MockTelephonyTransport,
)

__all__ = [
    # Interfaces
    "TelephonyTransport",
    "CallSessionRepository",
    # Models & Telemetry
    "TelephonyEvent",
    "TelephonyEventType",
    "CallSummary",
    "CallObservabilityDiagnostics",
    # Audio Conversion
    "TelephonyAudioConverter",
    # Transports
    "MockTelephonyTransport",
    "GenericWebSocketTelephonyTransport",
    # Repositories
    "MockCallSessionRepository",
    # Adapter
    "TelephonyVoiceAdapter",
]
