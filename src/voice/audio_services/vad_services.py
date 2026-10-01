"""Silero Voice Activity Detection (VAD) Service.

Provides real-time speech boundary detection, pre-roll speech buffering,
hysteresis filtering, and speech duration tracking.
"""

from __future__ import annotations

import collections
from enum import Enum
import os
from pathlib import Path
import time
from typing import List, NamedTuple, Optional
import warnings
import numpy as np
import torch

from src.logging import get_logger

logger = get_logger("voice.vad_service")


class VADEventType(str, Enum):
    NONE = "none"
    START_OF_SPEECH = "start_of_speech"
    ACTIVE_SPEECH = "active_speech"
    END_OF_SPEECH = "end_of_speech"


class VADEvent(NamedTuple):
    event_type: VADEventType
    audio_pcm16: Optional[bytes] = None
    speech_probability: float = 0.0
    duration_ms: float = 0.0


class VADService:
    """Silero VAD real-time streaming engine with hysteresis and pre-roll preservation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        start_threshold: float = 0.45,
        end_threshold: float = 0.20,
        min_speech_duration_ms: int = 160,
        min_silence_duration_ms: int = 400,
        pre_roll_duration_ms: int = 300,
    ):
        self.sample_rate = sample_rate
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold

        self.frame_samples = 512  # Exactly 32ms at 16kHz
        self.frame_bytes = self.frame_samples * 2

        self.min_speech_frames = max(1, int(min_speech_duration_ms / 32))
        self.min_silence_frames = max(1, int(min_silence_duration_ms / 32))
        self.pre_roll_frames = max(1, int(pre_roll_duration_ms / 32))

        self.pre_roll_buffer = collections.deque(maxlen=self.pre_roll_frames)
        self.is_speaking = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.collected_speech_frames: List[bytes] = []
        self.collected_speech_probs: List[float] = []

        self.raw_buffer = bytearray()
        self._debug_counter = 0

        self.model = None
        self._init_model()
        self.reset()

    def _init_model(self) -> None:
        """Load Silero VAD neural model with multiple fallback strategies."""
        if os.getenv("FORCE_MOCK", "").lower() == "true":
            logger.info("FORCE_MOCK is set: using energy-based model for deterministic testing")
            self.model = None
            return

        # 1. Try local torch hub cache JIT model directly (bypasses torchaudio requirement)
        cache_candidates = [
            Path(os.path.expanduser("~")) / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master" / "src" / "silero_vad" / "data" / "silero_vad.jit",
            Path(__file__).resolve().parent.parent.parent.parent / "models" / "silero_vad.jit",
        ]
        for jit_path in cache_candidates:
            if jit_path.exists():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.model = torch.jit.load(str(jit_path))
                    self.model.eval()
                    logger.info("Loaded Silero VAD JIT model from %s", jit_path)
                    return
                except Exception as e:
                    logger.warning("Failed loading JIT from %s: %s", jit_path, e)

        # 2. Try torch.hub.load
        try:
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.model.eval()
            logger.info("Loaded Silero VAD via torch.hub")
            return
        except Exception as e:
            logger.warning("torch.hub Silero VAD load failed: %s", e)

        # 3. Energy-based fallback model
        logger.info("Using energy-based fallback for VAD")
        self.model = None

    def reset(self) -> None:
        """Clear audio buffers and reset recurrent neural states."""
        self.raw_buffer.clear()
        self.pre_roll_buffer.clear()
        self.collected_speech_frames.clear()
        self.collected_speech_probs.clear()
        self.is_speaking = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        if self.model is not None and hasattr(self.model, "reset_states"):
            try:
                self.model.reset_states()
            except Exception:
                pass

    def _compute_speech_prob(self, audio_np: np.ndarray) -> float:
        """Compute speech probability using PyTorch model or energy fallback."""
        if self.model is not None:
            try:
                tensor_chunk = torch.from_numpy(audio_np).unsqueeze(0)
                with torch.no_grad():
                    prob = self.model(tensor_chunk, self.sample_rate)
                    return float(prob.squeeze().item())
            except Exception as e:
                logger.debug("Neural VAD inference error: %s", e)

        # Energy fallback
        rms = np.sqrt(np.mean(audio_np ** 2)) if len(audio_np) > 0 else 0.0
        return 0.85 if rms > 0.015 else 0.05

    def process(self, pcm_bytes: bytes) -> List[VADEvent]:
        """Process incoming raw PCM16 bytes and emit stateful VAD events."""
        events: List[VADEvent] = []
        if not pcm_bytes:
            return events

        self.raw_buffer.extend(pcm_bytes)

        while len(self.raw_buffer) >= self.frame_bytes:
            frame = bytes(self.raw_buffer[:self.frame_bytes])
            del self.raw_buffer[:self.frame_bytes]

            audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            speech_prob = self._compute_speech_prob(audio_np)

            self._debug_counter += 1
            if self._debug_counter % 30 == 0:
                max_amp = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
                logger.debug(
                    "[VAD Meter] Peak Amp: %.3f | Speech Prob: %.3f | Speaking: %s",
                    max_amp, speech_prob, self.is_speaking
                )

            if not self.is_speaking:
                self.pre_roll_buffer.append(frame)

                if speech_prob >= self.start_threshold:
                    self.speech_frames_count += 1
                    if self.speech_frames_count >= self.min_speech_frames:
                        self.is_speaking = True
                        self.silence_frames_count = 0
                        self.speech_frames_count = 0

                        # Prepend preserved pre-roll buffer to prevent cutting off initial consonants
                        self.collected_speech_frames = list(self.pre_roll_buffer)
                        self.collected_speech_probs = [speech_prob] * len(self.pre_roll_buffer)
                        self.pre_roll_buffer.clear()

                        events.append(
                            VADEvent(
                                event_type=VADEventType.START_OF_SPEECH,
                                speech_probability=speech_prob,
                            )
                        )
                else:
                    self.speech_frames_count = 0

            else:
                self.collected_speech_frames.append(frame)
                self.collected_speech_probs.append(speech_prob)

                if speech_prob < self.end_threshold:
                    self.silence_frames_count += 1
                else:
                    self.silence_frames_count = 0

                if self.silence_frames_count >= self.min_silence_frames:
                    prune_count = self.min_silence_frames
                    speech_frames = (
                        self.collected_speech_frames[:-prune_count]
                        if len(self.collected_speech_frames) > prune_count
                        else self.collected_speech_frames
                    )
                    speech_probs = (
                        self.collected_speech_probs[:-prune_count]
                        if len(self.collected_speech_probs) > prune_count
                        else self.collected_speech_probs
                    )

                    complete_audio = b"".join(speech_frames)
                    duration_ms = (len(complete_audio) / 2 / self.sample_rate) * 1000
                    avg_prob = float(np.mean(speech_probs)) if speech_probs else speech_prob

                    events.append(
                        VADEvent(
                            event_type=VADEventType.END_OF_SPEECH,
                            audio_pcm16=complete_audio,
                            speech_probability=avg_prob,
                            duration_ms=duration_ms,
                        )
                    )

                    self.is_speaking = False
                    self.silence_frames_count = 0
                    self.collected_speech_frames.clear()
                    self.collected_speech_probs.clear()
                    if self.model is not None and hasattr(self.model, "reset_states"):
                        try:
                            self.model.reset_states()
                        except Exception:
                            pass
                else:
                    events.append(
                        VADEvent(
                            event_type=VADEventType.ACTIVE_SPEECH,
                            speech_probability=speech_prob,
                        )
                    )

        return events


SileroVADService = VADService
