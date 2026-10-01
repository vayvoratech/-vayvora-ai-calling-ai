"""Dependency injection providers for FastAPI routers."""

from __future__ import annotations

import os
from typing import Optional
from src.core.engine import ConversationEngine
from src.state.manager import ConversationStateManager
from src.ui.bootstrap import bootstrap_workbench
from src.ui.service import WorkbenchService
from src.voice.session_manager import VoiceSessionManager
from src.voice.stt.stt_service import GroqWhisperSTT, STTService
from src.voice.tts.deepgram_tts_service import DeepgramFluxTTS, DeepgramTTS

_workbench_service: Optional[WorkbenchService] = None
_voice_session_manager: Optional[VoiceSessionManager] = None
_stt_service: Optional[STTService] = None
_tts_service: Optional[DeepgramFluxTTS] = None


def get_workbench_service() -> WorkbenchService:
    """Return initialized WorkbenchService singleton."""
    global _workbench_service
    if _workbench_service is None:
        force_mock = os.getenv("FORCE_MOCK", "false").lower() in ("true", "1", "yes")
        _workbench_service = bootstrap_workbench(force_mock=force_mock)
    return _workbench_service


def get_conversation_engine() -> ConversationEngine:
    """Return the application's central ConversationEngine instance."""
    service = get_workbench_service()
    return service.engine


def get_state_manager() -> ConversationStateManager:
    """Return the application's central ConversationStateManager instance."""
    service = get_workbench_service()
    return service.state_manager


def get_stt_service() -> STTService:
    """Return the STTService singleton."""
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service


def get_tts_service() -> DeepgramFluxTTS:
    """Return the DeepgramFluxTTS singleton."""
    global _tts_service
    if _tts_service is None:
        _tts_service = DeepgramFluxTTS()
    return _tts_service


def get_voice_session_manager() -> VoiceSessionManager:
    """Return the VoiceSessionManager singleton."""
    global _voice_session_manager
    if _voice_session_manager is None:
        engine = get_conversation_engine()
        state_mgr = get_state_manager()
        stt = get_stt_service()
        tts = get_tts_service()
        _voice_session_manager = VoiceSessionManager(
            engine=engine,
            state_manager=state_mgr,
            stt_service=stt,
            tts_service=tts,
        )
    return _voice_session_manager
