from __future__ import annotations

import math
from typing import Any, Dict, Optional
import numpy as np

from src.logging import get_logger

logger = get_logger("voice.noise_suppression")


class NoiseSuppressor:
    """
    Production-grade real-time noise suppression and acoustic hygiene pipeline.

    Multi-stage architecture:
    1. High-Pass Filter (sub-80Hz rumble, electrical hum, mic pops)
    2. WebRTC Acoustic Processing Model (APM) Noise Suppression (stationary noise, fan, AC)
    3. Adaptive Soft Noise Gate (suppresses residual room hiss during speech pauses without clipping)
    4. Real-time DSP Telemetry (RMS, attenuation dB, WebRTC speech probability)

    Zero-buffering-latency design: processes arbitrary chunk sizes instantaneously
    while preserving 1:1 input/output sample count alignment.
    Thread-safe and session-isolated: each instance maintains independent DSP filter state.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        enabled: bool = True,
        level: int = 2,
        high_pass_filter: bool = True,
        echo_cancellation: bool = False,
        auto_gain_control: bool = False,
        noise_gate_enabled: bool = True,
        noise_gate_threshold_db: float = -42.0,
        noise_gate_attenuation_db: float = 15.0,
        noise_gate_attack_ms: float = 10.0,
        noise_gate_release_ms: float = 60.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.level = level
        self.high_pass_filter = high_pass_filter
        self.echo_cancellation = echo_cancellation
        self.auto_gain_control = auto_gain_control

        # Noise Gate Configuration
        self.noise_gate_enabled = noise_gate_enabled
        self.noise_gate_threshold_db = noise_gate_threshold_db
        self.noise_gate_attenuation_db = noise_gate_attenuation_db
        self.noise_gate_attack_ms = max(1.0, noise_gate_attack_ms)
        self.noise_gate_release_ms = max(5.0, noise_gate_release_ms)

        # Gate slice duration: ~10ms slices for gain computation
        self.slice_samples = int(sample_rate * 0.010)  # 160 samples at 16kHz
        dt_sec = self.slice_samples / float(sample_rate)
        self._alpha_attack = 1.0 - math.exp(-dt_sec / (self.noise_gate_attack_ms / 1000.0))
        self._alpha_release = 1.0 - math.exp(-dt_sec / (self.noise_gate_release_ms / 1000.0))
        self._min_gate_gain = 10.0 ** (-abs(self.noise_gate_attenuation_db) / 20.0)
        self._current_gate_gain = 1.0

        # Buffer for incomplete odd bytes (PCM16 requires 2 bytes per sample)
        self._odd_byte_buffer = b""

        # Telemetry
        self._total_input_energy = 0.0
        self._total_output_energy = 0.0
        self._total_samples_processed = 0
        self._last_speech_prob = 0.0
        self._last_input_rms = 0.0
        self._last_output_rms = 0.0

        self.processor = None

        if not enabled:
            logger.info("Noise suppression disabled by configuration")
            return

        try:
            from pywebrtc_audio import AudioProcessor as WebRTCAudioProcessor

            self.processor = WebRTCAudioProcessor(
                sample_rate=sample_rate,
                num_channels=1,
                echo_cancellation=echo_cancellation,
                noise_suppression=True,
                high_pass_filter=high_pass_filter,
                auto_gain_control=auto_gain_control,
                ns_level=level,
            )

            logger.info(
                "WebRTC noise suppression initialized: "
                "sample_rate=%dHz level=%d high_pass=%s gate=%s (thresh=%.1fdB)",
                sample_rate,
                level,
                high_pass_filter,
                noise_gate_enabled,
                noise_gate_threshold_db,
            )

        except Exception as exc:
            logger.warning(
                "Failed to initialize pywebrtc_audio AudioProcessor: %s. "
                "Noise suppression will operate in pass-through mode.",
                exc,
            )
            self.processor = None

    def process(self, pcm16_bytes: bytes) -> bytes:
        """
        Apply real-time noise suppression, high-pass filtering, and soft gating.

        Preserves exact 1:1 input/output sample count without introducing frame-delay.
        Handles odd-byte alignment gracefully.
        """
        if not pcm16_bytes and not self._odd_byte_buffer:
            return b""

        # Combine with leftover odd byte from previous chunk if present
        if self._odd_byte_buffer:
            pcm16_bytes = self._odd_byte_buffer + (pcm16_bytes or b"")
            self._odd_byte_buffer = b""

        if not pcm16_bytes:
            return b""

        # PCM16 requires even number of bytes (2 bytes per sample)
        if len(pcm16_bytes) % 2 != 0:
            self._odd_byte_buffer = pcm16_bytes[-1:]
            pcm16_bytes = pcm16_bytes[:-1]

        if not pcm16_bytes:
            return b""

        # Pass-through if noise suppression is disabled
        if not self.enabled:
            return pcm16_bytes

        audio_in = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if audio_in.size == 0:
            return b""

        # Telemetry: Track input energy
        in_rms = float(np.sqrt(np.mean(audio_in.astype(np.float64) ** 2)))
        self._last_input_rms = in_rms
        self._total_input_energy += (in_rms ** 2) * audio_in.size
        self._total_samples_processed += audio_in.size

        # Stage 1: WebRTC APM (High-Pass + Noise Suppressor)
        if self.processor is not None:
            try:
                processed = self.processor.process(audio_in)
                processed_np = np.asarray(processed)
                if hasattr(self.processor, "speech_probability"):
                    self._last_speech_prob = float(self.processor.speech_probability)
            except Exception as exc:
                logger.debug("WebRTC process frame error: %s", exc)
                processed_np = audio_in
        else:
            processed_np = audio_in

        # Stage 2: Adaptive Soft Noise Gate
        if self.noise_gate_enabled and processed_np.size > 0:
            gated_out = np.empty_like(processed_np, dtype=np.float64)
            slice_len = self.slice_samples

            for i in range(0, processed_np.size, slice_len):
                chunk = processed_np[i : i + slice_len]
                chunk_rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                chunk_dbfs = 20.0 * np.log10(max(chunk_rms, 1e-5) / 32768.0)

                # Target gain calculation
                if chunk_dbfs >= self.noise_gate_threshold_db:
                    target_gain = 1.0
                    alpha = self._alpha_attack
                else:
                    target_gain = self._min_gate_gain
                    alpha = self._alpha_release

                # Smooth exponential gain transition (no audio pops/clicks)
                self._current_gate_gain += alpha * (target_gain - self._current_gate_gain)

                # Apply gain to this slice
                gated_out[i : i + chunk.size] = chunk.astype(np.float64) * self._current_gate_gain

            processed_np = np.clip(gated_out, -32768, 32767).astype(np.int16)

        out_rms = float(np.sqrt(np.mean(processed_np.astype(np.float64) ** 2)))
        self._last_output_rms = out_rms
        self._total_output_energy += (out_rms ** 2) * processed_np.size

        return processed_np.tobytes()

    def flush(self) -> bytes:
        """Flush any remaining buffered odd bytes."""
        if not self._odd_byte_buffer:
            return b""
        # Pad 1 zero byte to complete 1 sample
        padded = self._odd_byte_buffer + b"\x00"
        self._odd_byte_buffer = b""
        return self.process(padded)

    @property
    def speech_probability(self) -> float:
        """Current WebRTC APM speech probability (0.0 to 1.0)."""
        return self._last_speech_prob

    @property
    def is_gate_open(self) -> bool:
        """True if the soft noise gate is currently admitting speech."""
        return self._current_gate_gain > 0.5

    def get_telemetry(self) -> Dict[str, Any]:
        """Return running DSP metrics for acoustic diagnostics and monitoring."""
        in_rms = (
            math.sqrt(self._total_input_energy / max(1, self._total_samples_processed))
            if self._total_samples_processed > 0
            else 0.0
        )
        out_rms = (
            math.sqrt(self._total_output_energy / max(1, self._total_samples_processed))
            if self._total_samples_processed > 0
            else 0.0
        )
        attenuation_db = (
            20.0 * math.log10(max(in_rms, 1e-4) / max(out_rms, 1e-4))
            if in_rms > 0 and out_rms > 0
            else 0.0
        )
        return {
            "enabled": self.enabled,
            "ns_level": self.level,
            "high_pass": self.high_pass_filter,
            "last_input_rms": round(self._last_input_rms, 2),
            "last_output_rms": round(self._last_output_rms, 2),
            "attenuation_db": round(attenuation_db, 2),
            "speech_probability": round(self._last_speech_prob, 4),
            "gate_gain": round(self._current_gate_gain, 4),
            "is_gate_open": self.is_gate_open,
        }

    def reset(self) -> None:
        """Reset internal odd byte buffer, WebRTC state, and gain filters for a clean session start."""
        self._odd_byte_buffer = b""
        self._current_gate_gain = 1.0
        self._total_input_energy = 0.0
        self._total_output_energy = 0.0
        self._total_samples_processed = 0
        self._last_speech_prob = 0.0
        self._last_input_rms = 0.0
        self._last_output_rms = 0.0

        if self.processor is not None:
            reset_fn = getattr(self.processor, "reset", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception as exc:
                    logger.debug("WebRTC processor reset error: %s", exc)

    def close(self) -> None:
        """Release underlying WebRTC APM native resources."""
        self.reset()
        if self.processor is not None:
            close_fn = getattr(self.processor, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self.processor = None