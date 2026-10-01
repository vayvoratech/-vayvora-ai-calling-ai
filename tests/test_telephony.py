"""Unit tests for Phase 9: Telephony Transport Layer and Adapters.

Tests transport connections, packet validation, audio resampling,
repository persistence, and call summary extraction.
"""

import pytest
import asyncio
import time

from src.core.types import DomainType, CallDirection, ConversationStage, ToolExecutionResult
from src.core.errors import TelephonyAudioIncompatibleError, TelephonyConnectionError, TelephonyPacketError
from src.state.models import ConversationState, CallMetadata, CallerProfile
from src.audio.schemas import AudioChunk
from src.telephony.models import (
    TelephonyEventType,
    TelephonyEvent,
    CallSummary,
    CallObservabilityDiagnostics,
)
from src.telephony.audio import TelephonyAudioConverter
from src.telephony.transport import MockTelephonyTransport, GenericWebSocketTelephonyTransport
from src.telephony.repository import MockCallSessionRepository
from src.telephony.adapter import TelephonyVoiceAdapter, TelephonyAudioOutputSink
from src.state.manager import ConversationStateManager
from src.audio.vad import MockVADProvider
from src.audio.stt import MockSTTProvider
from src.audio.tts import MockTTSProvider
from src.core.engine import ConversationEngine
from src.core.llm import MockLLMProvider
from src.rag.embeddings import MockEmbeddingProvider
from src.rag.retriever import GroundedKnowledgeProvider
from src.tools.mcp_client import MockToolProvider


# -----------------------------------------------------------------------------
# 1. Transport Lifecycle & Buffering Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_transport_connect_disconnect():
    """Verify MockTelephonyTransport connection state transitions."""
    transport = MockTelephonyTransport(
        call_id="call-001",
        session_id="session-001",
        direction=CallDirection.INBOUND,
        caller_number="+15550001",
    )
    assert not transport.is_connected

    await transport.connect()
    assert transport.is_connected

    # Sending audio when connected
    pcm_frame = b"\x00\x01" * 160  # 320 bytes = 20ms of 8kHz 16-bit
    await transport.send_audio(pcm_frame)
    assert len(transport.sent_audio_frames) == 1
    assert transport.sent_audio_frames[0] == pcm_frame

    # Disconnect
    await transport.disconnect()
    assert not transport.is_connected

    # Sending after disconnect should raise TelephonyConnectionError
    with pytest.raises(TelephonyConnectionError):
        await transport.send_audio(pcm_frame)


@pytest.mark.asyncio
async def test_mock_transport_incoming_audio_queue():
    """Verify incoming audio queuing and FIFO retrieval."""
    transport = MockTelephonyTransport(
        call_id="call-002",
        session_id="session-002",
        direction=CallDirection.INBOUND,
    )
    await transport.connect()

    packet_a = b"\x10\x20" * 80
    packet_b = b"\x30\x40" * 80

    transport.feed_audio(packet_a)
    transport.feed_audio(packet_b)

    rec_a = await transport.receive_audio()
    assert rec_a == packet_a

    rec_b = await transport.receive_audio()
    assert rec_b == packet_b


@pytest.mark.asyncio
async def test_mock_transport_events():
    """Verify event emission and recording."""
    transport = MockTelephonyTransport(call_id="call-003", session_id="session-003")
    await transport.connect()

    event = TelephonyEvent(
        event_type=TelephonyEventType.MEDIA_STARTED,
        call_id="call-003",
        payload={"seq": 1},
    )
    await transport.send_event(event)

    assert len(transport.sent_events) == 1
    assert transport.sent_events[0].event_type == TelephonyEventType.MEDIA_STARTED
    assert transport.sent_events[0].payload["seq"] == 1


# -----------------------------------------------------------------------------
# 2. Telephony Audio Converter Tests
# -----------------------------------------------------------------------------


def test_converter_validation_valid():
    """Verify converter accepts standard 16-bit PCM byte packets."""
    converter = TelephonyAudioConverter(target_sample_rate=16000)
    valid_pcm = b"\x00\x00" * 160  # 320 bytes = 20ms at 8kHz 16-bit mono
    chunk = converter.telephony_frame_to_chunk(valid_pcm, sample_rate=8000)
    assert chunk.sample_rate == 16000
    assert len(chunk.data) == 640  # upsampled from 160 to 320 samples


def test_converter_validation_invalid_odd_bytes():
    """Verify converter rejects odd byte lengths for 16-bit PCM."""
    converter = TelephonyAudioConverter(target_sample_rate=16000)
    invalid_pcm = b"\x00" * 161  # Odd length cannot be 16-bit samples
    with pytest.raises(TelephonyPacketError):
        converter.telephony_frame_to_chunk(invalid_pcm, sample_rate=8000)


def test_converter_validation_empty():
    """Verify converter rejects empty audio packets."""
    converter = TelephonyAudioConverter()
    with pytest.raises(TelephonyPacketError):
        converter.telephony_frame_to_chunk(b"", sample_rate=8000)


def test_converter_validation_incompatible_encoding():
    """Verify converter rejects unsupported encodings."""
    converter = TelephonyAudioConverter()
    valid_pcm = b"\x00\x00" * 80
    with pytest.raises(TelephonyAudioIncompatibleError):
        converter.telephony_frame_to_chunk(valid_pcm, sample_rate=8000, encoding="mp3")


def test_converter_telephony_to_pipeline_upsampling():
    """Verify 8kHz -> 16kHz upsampling doubles sample count."""
    converter = TelephonyAudioConverter(target_sample_rate=16000)
    # 80 samples at 8kHz = 160 bytes
    in_pcm = b"\x10\x00" * 80
    chunk = converter.telephony_frame_to_chunk(in_pcm, sample_rate=8000)

    assert isinstance(chunk, AudioChunk)
    assert chunk.sample_rate == 16000
    # 80 samples upsampled 2x should yield 160 samples = 320 bytes
    assert len(chunk.data) == 320


def test_converter_pipeline_to_telephony_downsampling():
    """Verify 16kHz -> 8kHz downsampling halves sample count and strips RIFF header."""
    converter = TelephonyAudioConverter(target_sample_rate=16000)

    # 160 samples at 16kHz = 320 bytes raw PCM
    raw_pcm = b"\x20\x00" * 160
    chunk = AudioChunk(data=raw_pcm, sample_rate=16000, duration=0.01, format="pcm16")

    telephony_packet = converter.chunk_to_telephony_frame(chunk, dst_sample_rate=8000)
    # Downsampled by 2: 80 samples = 160 bytes
    assert len(telephony_packet) == 160

    # Test with dummy RIFF WAV header prepended
    wav_with_header = b"RIFF" + (b"\x00" * 40) + raw_pcm
    chunk_wav = AudioChunk(data=wav_with_header, sample_rate=16000, duration=0.01, format="wav")
    stripped_packet = converter.chunk_to_telephony_frame(chunk_wav, dst_sample_rate=8000)
    assert len(stripped_packet) == 160


# -----------------------------------------------------------------------------
# 3. Call Session Repository Tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_call_session_repository_crud():
    """Verify repository persistence, lookup, listing, and deletion."""
    repo = MockCallSessionRepository()

    summary = CallSummary(
        call_id="call-rep-01",
        session_id="session-rep-01",
        domain="edusaas",
        direction="inbound",
        caller={"phone": "+15559876"},
        intent_history=["greeting", "curriculum_query", "farewell"],
        business_status="inquiry_answered",
        termination_reason="completed_goodbye",
        start_time=100.0,
        end_time=142.5,
        duration=42.5,
    )

    await repo.persist_summary(summary)

    # Fetch
    fetched = await repo.get_summary("call-rep-01")
    assert fetched is not None
    assert fetched.call_id == "call-rep-01"
    assert fetched.domain == "edusaas"
    assert fetched.duration == 42.5
    assert fetched.termination_reason == "completed_goodbye"

    # List
    all_summaries = await repo.list_summaries()
    assert len(all_summaries) == 1
    assert all_summaries[0].call_id == "call-rep-01"

    # Delete
    repo.clear()
    assert await repo.get_summary("call-rep-01") is None


# -----------------------------------------------------------------------------
# 4. CallSummary Factory & Contract Integrity Tests
# -----------------------------------------------------------------------------


def test_call_summary_from_conversation_state():
    """Verify CallSummary sanitizes data and captures verified tools."""
    state_mgr = ConversationStateManager()
    state = state_mgr.create_inbound_state(
        call_id="call-fact-01",
        caller_phone="+15554321",
        domain=DomainType.EDUSAAS,
        caller_name="Bob User",
    )

    # Simulate completed verified action
    state.last_action = "send_email"
    state.last_tool_result = ToolExecutionResult(
        tool_name="send_email",
        success=True,
        verification_code="em-123",
        data={"status": "sent"},
    )

    summary = CallSummary.from_conversation_state(
        state=state,
        session_id="voice-session-fact-01",
        start_time=10.0,
        end_time=25.0,
        disconnect_reason="remote_caller_disconnected",
    )

    assert summary.call_id == "call-fact-01"
    assert summary.domain == "edusaas"
    assert summary.direction == "inbound"
    assert summary.caller["phone"] == "+15554321"
    assert summary.termination_reason == "remote_caller_disconnected"
    assert summary.duration == 15.0
    # Verified action captured
    assert len(summary.completed_actions) == 1
    assert summary.completed_actions[0]["action"] == "send_email"
    assert summary.completed_actions[0]["verified"] is True
    assert summary.completed_actions[0]["reference"] == "em-123"


# -----------------------------------------------------------------------------
# 5. TelephonyVoiceAdapter Integration Tests
# -----------------------------------------------------------------------------


def build_test_adapter(direction: CallDirection = CallDirection.INBOUND):
    """Helper to assemble adapter with mocked components."""
    transport = MockTelephonyTransport(
        call_id="call-adapt-01",
        session_id="session-adapt-01",
        direction=direction,
        caller_number="+15553333",
    )
    state_mgr = ConversationStateManager()
    vad = MockVADProvider(sample_rate=16000)
    stt = MockSTTProvider()
    tts = MockTTSProvider(sample_rate=16000)
    llm = MockLLMProvider()
    rag = GroundedKnowledgeProvider(
        embedding_provider=MockEmbeddingProvider(dimension=64),
        in_memory=True,
    )
    mcp = MockToolProvider()
    engine = ConversationEngine(
        llm_provider=llm,
        knowledge_provider=rag,
        tool_provider=mcp,
    )
    repo = MockCallSessionRepository()

    adapter = TelephonyVoiceAdapter(
        transport=transport,
        state_manager=state_mgr,
        vad_provider=vad,
        stt_provider=stt,
        conversation_engine=engine,
        tts_provider=tts,
        repository=repo,
    )
    return adapter, repo, transport, stt


@pytest.mark.asyncio
async def test_telephony_voice_adapter_inbound_lifecycle():
    """Verify full inbound connection, packet ingestion, and disconnect with summary persistence."""
    adapter, repo, transport, stt = build_test_adapter(CallDirection.INBOUND)

    # 1. Inbound Call
    session = await adapter.handle_inbound_call(domain=DomainType.EDUSAAS)
    assert transport.is_connected
    assert session.is_active
    assert adapter.conversation_state.current_domain == DomainType.EDUSAAS

    # 2. Process Telephony Packet (Speech)
    stt.add_transcription("Tell me about machine learning.")
    pcm_audio = b"\x01\x00" * 320  # 20ms at 8kHz 16-bit
    res = await adapter.process_incoming_packet(pcm_audio, sample_rate=8000)
    assert res is not None

    # Check observability
    assert adapter.observability.first_audio_time is not None

    # 3. Disconnect
    summary = await adapter.handle_disconnect(reason="remote_caller_disconnected")
    assert not transport.is_connected
    assert not session.is_active
    assert summary.termination_reason == "remote_caller_disconnected"

    # Verify persisted in repository
    saved_summary = await repo.get_summary(transport.call_id)
    assert saved_summary is not None
    assert saved_summary.call_id == transport.call_id


@pytest.mark.asyncio
async def test_telephony_voice_adapter_outbound_lifecycle():
    """Verify outbound call connection preserves campaign details and context."""
    adapter, repo, transport, stt = build_test_adapter(CallDirection.OUTBOUND)

    session = await adapter.handle_outbound_call(
        domain=DomainType.VAYVORA,
        caller_name="Sarah Connor",
        campaign_id="CAMP-AI-01",
        campaign_objective="Discuss enterprise AI orchestration platform",
    )
    assert transport.is_connected
    assert session.is_active
    assert adapter.conversation_state.current_domain == DomainType.VAYVORA
    assert adapter.conversation_state.metadata.direction == CallDirection.OUTBOUND
    assert adapter.conversation_state.caller.name == "Sarah Connor"
    assert adapter.conversation_state.metadata.campaign_id == "CAMP-AI-01"
    assert adapter.conversation_state.caller.campaign == "CAMP-AI-01"
    assert adapter.conversation_state.caller.campaign_objective == "Discuss enterprise AI orchestration platform"

    summary = await adapter.handle_disconnect(reason="completed_goodbye")
    assert summary.termination_reason == "completed_goodbye"
