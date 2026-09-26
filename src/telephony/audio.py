"""Telephony Audio Framing, Format Validation, and Conversion Utilities.

Bridges telephony media packets (RTP/G.711/PCM) to application AudioChunks,
validating audio specifications and enforcing strict compatibility checks.
"""

from typing import Optional
import numpy as np

from src.audio.schemas import AudioChunk
from src.core.errors import TelephonyAudioIncompatibleError, TelephonyPacketError
from src.logging import get_logger

logger = get_logger("telephony.audio")


class TelephonyAudioConverter:
    """Validates and converts telephony audio streams to and from AudioChunk models."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        sample_width: int = 2,  # 16-bit PCM = 2 bytes per sample
    ) -> None:
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.sample_width = sample_width

    def telephony_frame_to_chunk(
        self,
        raw_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        timestamp: float = 0.0,
        encoding: str = "pcm16",
    ) -> AudioChunk:
        """Validate and convert an incoming telephony audio packet to AudioChunk."""
        if not raw_bytes or len(raw_bytes) == 0:
            raise TelephonyPacketError("Received empty telephony audio packet.")

        # Encoding validation
        if encoding.lower() not in ["pcm16", "linear16", "raw_pcm"]:
            raise TelephonyAudioIncompatibleError(
                f"Unsupported telephony encoding: '{encoding}'. Expected 16-bit PCM."
            )

        # Channel validation
        if channels != 1 and channels != 2:
            raise TelephonyAudioIncompatibleError(
                f"Unsupported channel count: {channels}. Must be 1 (mono) or 2 (stereo)."
            )

        # Sample rate validation
        if sample_rate < 8000 or sample_rate > 48000:
            raise TelephonyAudioIncompatibleError(
                f"Invalid telephony sample rate: {sample_rate} Hz (must be 8kHz - 48kHz)."
            )

        # Sample width / frame boundary validation (must be multiple of sample_width * channels)
        bytes_per_sample = self.sample_width * channels
        if len(raw_bytes) % bytes_per_sample != 0:
            raise TelephonyPacketError(
                f"Malformed audio packet: length {len(raw_bytes)} bytes is not a multiple of "
                f"sample width {bytes_per_sample}."
            )

        # Resample to pipeline sample rate (16kHz) if received as narrowband (8kHz)
        final_bytes = raw_bytes
        final_sample_rate = sample_rate

        if sample_rate != self.target_sample_rate:
            final_bytes = self.resample_pcm16(
                audio_bytes=raw_bytes,
                src_rate=sample_rate,
                dst_rate=self.target_sample_rate,
                channels=channels,
            )
            final_sample_rate = self.target_sample_rate

        duration = len(final_bytes) / (final_sample_rate * self.sample_width * channels)

        return AudioChunk(
            data=final_bytes,
            sample_rate=final_sample_rate,
            channels=channels,
            format="pcm16",
            duration=round(duration, 4),
            timestamp=round(timestamp, 4),
        )

    def chunk_to_telephony_frame(
        self,
        chunk: AudioChunk,
        dst_sample_rate: Optional[int] = None,
    ) -> bytes:
        """Convert an application AudioChunk or WAV payload into telephony PCM frame."""
        raw_data = chunk.data
        src_rate = chunk.sample_rate

        # If data starts with RIFF header, strip the 44-byte WAV container
        if raw_data.startswith(b"RIFF") and len(raw_data) > 44:
            raw_data = raw_data[44:]

        target_rate = dst_sample_rate or self.target_sample_rate
        if src_rate != target_rate:
            raw_data = self.resample_pcm16(
                audio_bytes=raw_data,
                src_rate=src_rate,
                dst_rate=target_rate,
                channels=chunk.channels,
            )

        return raw_data

    @staticmethod
    def resample_pcm16(
        audio_bytes: bytes,
        src_rate: int,
        dst_rate: int,
        channels: int = 1,
    ) -> bytes:
        """Deterministic linear PCM16 resampling using numpy."""
        if src_rate == dst_rate or len(audio_bytes) < 2:
            return audio_bytes

        try:
            pcm_in = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(pcm_in) == 0:
                return audio_bytes

            num_output_samples = int(len(pcm_in) * (float(dst_rate) / float(src_rate)))
            if num_output_samples <= 0:
                return b""

            indices = np.linspace(0, len(pcm_in) - 1, num_output_samples)
            pcm_out = np.interp(indices, np.arange(len(pcm_in)), pcm_in).astype(np.int16)
            return pcm_out.tobytes()
        except Exception as exc:
            logger.warning("Resampling failed: %s, returning original audio.", exc)
            return audio_bytes
