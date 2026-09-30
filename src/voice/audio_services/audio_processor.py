"""Audio preprocessing, multi-stage noise suppression, resampling, and VAD integration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.signal import resample_poly

from src.logging import get_logger
from src.voice.audio_services.noise_suppression import NoiseSuppressor
from src.voice.audio_services.vad_services import VADEventType, VADService

logger = get_logger("voice.audio_processor")


class AudioProcessor:
    """
    Streaming audio processor with multi-stage noise suppression and acoustic hygiene.

    Pipeline:
        Input PCM16 (e.g. 48kHz from browser or telephony)
            ↓
        Polyphase Rational Resampler (48kHz -> 16kHz)
            ↓
        WebRTC APM + High-Pass Filter (sub-80Hz rumble + stationary noise reduction)
            ↓
        FIFO Frame Chunking (10ms exact slices)
            ↓
        Soft Noise Gate (smooth exponential attenuation of residual room floor)
            ↓
        Silero VAD (160ms onset hysteresis + 300ms pre-roll preservation)
            ↓
        Pre-STT Acoustic Validation (duration, RMS energy, speech probability)
            ↓
        Clean Speech Audio Segment -> Groq Whisper STT
    """

    def __init__(
        self,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 16000,
        vad_start_threshold: float = 0.45,
        vad_end_threshold: float = 0.20,
        min_speech_duration_ms: int = 160,
        min_silence_duration_ms: int = 400,
        pre_roll_duration_ms: int = 300,
        noise_suppression_enabled: bool = True,
        noise_suppression_level: int = 2,
        high_pass_filter_enabled: bool = True,
        noise_gate_enabled: bool = True,
        noise_gate_threshold_db: float = -42.0,
        min_segment_duration_ms: int = 200,
        min_segment_rms: float = 120.0,
        min_segment_speech_prob: float = 0.35,
        echo_cancellation_enabled: bool = False,
        auto_gain_control_enabled: bool = False,
    ) -> None:
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

        self.min_segment_duration_ms = min_segment_duration_ms
        self.min_segment_rms = min_segment_rms
        self.min_segment_speech_prob = min_segment_speech_prob

        # ---------------------------------------------------------
        # Noise suppression (WebRTC APM + High-Pass + Soft Gate)
        # ---------------------------------------------------------
        self.noise_suppressor = NoiseSuppressor(
            sample_rate=output_sample_rate,
            enabled=noise_suppression_enabled,
            level=noise_suppression_level,
            high_pass_filter=high_pass_filter_enabled,
            echo_cancellation=echo_cancellation_enabled,
            auto_gain_control=auto_gain_control_enabled,
            noise_gate_enabled=noise_gate_enabled,
            noise_gate_threshold_db=noise_gate_threshold_db,
        )

        # ---------------------------------------------------------
        # Silero VAD
        # ---------------------------------------------------------
        self.vad = VADService(
            sample_rate=output_sample_rate,
            start_threshold=vad_start_threshold,
            end_threshold=vad_end_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            pre_roll_duration_ms=pre_roll_duration_ms,
        )

        logger.info(
            "AudioProcessor initialized: "
            "input=%dHz output=%dHz "
            "noise_suppression=%s level=%d high_pass=%s gate=%s (thresh=%.1fdB) "
            "vad_start=%.2f vad_end=%.2f min_speech=%dms silence=%dms pre_roll=%dms "
            "pre_stt_min_duration=%dms min_rms=%.1f min_prob=%.2f",
            input_sample_rate,
            output_sample_rate,
            noise_suppression_enabled,
            noise_suppression_level,
            high_pass_filter_enabled,
            noise_gate_enabled,
            noise_gate_threshold_db,
            vad_start_threshold,
            vad_end_threshold,
            min_speech_duration_ms,
            min_silence_duration_ms,
            pre_roll_duration_ms,
            min_segment_duration_ms,
            min_segment_rms,
            min_segment_speech_prob,
        )

    # ============================================================
    # RESAMPLING
    # ============================================================

    def resample(self, pcm16_bytes: bytes) -> bytes:
        """
        Resample PCM16 mono audio to the target sample rate.

        Example:
            48 kHz -> 16 kHz (3:1 rational downsampling)
        """
        if not pcm16_bytes:
            return b""

        if self.input_sample_rate == self.output_sample_rate:
            return pcm16_bytes

        # Ensure byte count aligns to 16-bit boundary (2 bytes per sample)
        if len(pcm16_bytes) % 2 != 0:
            pcm16_bytes = pcm16_bytes[:-1]

        if not pcm16_bytes:
            return b""

        audio_int16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if audio_int16.size == 0:
            return b""

        try:
            resampled = resample_poly(
                audio_int16,
                up=self.output_sample_rate,
                down=self.input_sample_rate,
            )
            resampled_int16 = np.clip(resampled, -32768, 32767).astype(np.int16)
            return resampled_int16.tobytes()

        except Exception:
            logger.exception("Audio resampling failed")
            raise

    def resample_48k_to_16k(self, pcm16_bytes: bytes) -> bytes:
        """Downsample 48 kHz PCM16 audio to 16 kHz."""
        return self.resample(pcm16_bytes)

    # ============================================================
    # AUDIO PREPROCESSING
    # ============================================================

    def preprocess(self, pcm16_bytes: bytes) -> bytes:
        """
        Prepare incoming audio for VAD/STT.

        Pipeline:
            PCM16 input -> Resample to 16 kHz -> Multi-stage Noise Suppression -> Clean PCM16
        """
        if not pcm16_bytes:
            return b""

        audio_16k = self.resample(pcm16_bytes)
        if not audio_16k:
            return b""

        clean_audio = self.noise_suppressor.process(audio_16k)
        return clean_audio

    # ============================================================
    # FRAME PROCESSING
    # ============================================================

    def process_audio(self, pcm16_bytes: bytes) -> List[bytes]:
        """
        Preprocess incoming audio and split it into 512-sample / 32 ms frames at 16 kHz.
        PCM16: 512 samples * 2 bytes = 1024 bytes.
        """
        processed = self.preprocess(pcm16_bytes)
        if not processed:
            return []

        frame_size = 1024
        frames: List[bytes] = []
        complete_length = len(processed) - (len(processed) % frame_size)

        for i in range(0, complete_length, frame_size):
            frames.append(processed[i : i + frame_size])

        return frames

    # ============================================================
    # VAD PROCESSING & PRE-STT VALIDATION
    # ============================================================

    def process_with_vad(self, pcm_bytes: bytes) -> Dict[str, Any]:
        """
        Preprocess audio through noise suppression and run Silero VAD with speech validation.

        Applies pre-STT validation on speech segments to reject non-speech noise spikes,
        transient clicks, and background audio before sending to STT.

        Returns:
            {
                "speech_started": bool,
                "speech_ended": bool,
                "is_speech": bool,
                "speech_audio": bytes | None,
                "duration_ms": float,
            }
        """
        if not pcm_bytes:
            return {
                "speech_started": False,
                "speech_ended": False,
                "is_speech": self.vad.is_speaking,
                "speech_audio": None,
                "duration_ms": 0.0,
            }

        # Step 1: Resample + Multi-stage noise suppression
        processed_audio = self.preprocess(pcm_bytes)
        if not processed_audio:
            return {
                "speech_started": False,
                "speech_ended": False,
                "is_speech": self.vad.is_speaking,
                "speech_audio": None,
                "duration_ms": 0.0,
            }

        # Step 2: Run Silero VAD on CLEAN audio
        vad_events = self.vad.process(processed_audio)

        result: Dict[str, Any] = {
            "speech_started": False,
            "speech_ended": False,
            "is_speech": self.vad.is_speaking,
            "speech_audio": None,
            "duration_ms": 0.0,
        }

        # Step 3: Process events & enforce Pre-STT acoustic validation
        for event in vad_events:
            if event.event_type == VADEventType.START_OF_SPEECH:
                result["speech_started"] = True
                result["is_speech"] = True
                logger.debug("VAD Speech started (onset prob: %.3f)", event.speech_probability)

            elif event.event_type == VADEventType.END_OF_SPEECH:
                speech_audio = event.audio_pcm16 or b""
                duration_ms = event.duration_ms
                speech_prob = event.speech_probability

                # Acoustic validation check to filter noise spikes
                is_valid_speech = True
                rejection_reason = ""

                # 1. Minimum duration check
                if duration_ms < self.min_segment_duration_ms:
                    is_valid_speech = False
                    rejection_reason = f"duration {duration_ms:.1f}ms < {self.min_segment_duration_ms}ms"

                # 2. Minimum RMS energy check
                if is_valid_speech and len(speech_audio) >= 2:
                    samples = np.frombuffer(speech_audio, dtype=np.int16)
                    segment_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    if segment_rms < self.min_segment_rms:
                        is_valid_speech = False
                        rejection_reason = f"RMS {segment_rms:.1f} < {self.min_segment_rms}"

                # 3. Minimum speech probability check
                if is_valid_speech and speech_prob < self.min_segment_speech_prob:
                    is_valid_speech = False
                    rejection_reason = f"speech prob {speech_prob:.3f} < {self.min_segment_speech_prob}"

                if is_valid_speech:
                    result["speech_ended"] = True
                    result["is_speech"] = False
                    result["speech_audio"] = speech_audio
                    result["duration_ms"] = duration_ms
                    logger.debug(
                        "VAD Speech ended & validated: duration=%.1fms, prob=%.3f",
                        duration_ms,
                        speech_prob,
                    )
                else:
                    logger.info("Filtered non-speech acoustic segment: %s", rejection_reason)
                    result["speech_ended"] = False
                    result["is_speech"] = False
                    result["speech_audio"] = None
                    result["duration_ms"] = 0.0

            elif event.event_type == VADEventType.ACTIVE_SPEECH:
                result["is_speech"] = True

        return result

    # ============================================================
    # TELEMETRY & DIAGNOSTICS
    # ============================================================

    def get_telemetry(self) -> Dict[str, Any]:
        """Return diagnostic metrics across noise suppression and VAD."""
        telemetry = self.noise_suppressor.get_telemetry()
        telemetry["is_speaking"] = self.vad.is_speaking
        telemetry["vad_start_threshold"] = self.vad.start_threshold
        telemetry["vad_end_threshold"] = self.vad.end_threshold
        return telemetry

    # ============================================================
    # RESET & CLOSE
    # ============================================================

    def reset(self) -> None:
        """Reset all DSP, FIFO buffers, and VAD neural recurrent state."""
        self.noise_suppressor.reset()
        self.vad.reset()
        logger.debug("AudioProcessor reset")

    def close(self) -> None:
        """Release audio processor DSP and VAD resources."""
        self.noise_suppressor.close()
        self.vad.reset()
        logger.debug("AudioProcessor closed")