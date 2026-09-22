from typing import Dict, Any, List
import numpy as np
from scipy.signal import resample_poly

from src.voice.audio_services.vad_services import VADService, VADEventType


class AudioProcessor:
    """
    Soniox-style Stream Audio Processor.
    
    Features:
      - Polyphase stream resampling (48kHz -> 16kHz) with edge continuity.
      - Maintains clean pipeline encapsulation.
      - Emits unified dictionary contracts for the WebSocket voice router.
    """

    def __init__(
        self,
        input_sample_rate: int = 48000,
        output_sample_rate: int = 16000,
        vad_start_threshold: float = 0.55,
        vad_end_threshold: float = 0.35,
        min_silence_duration_ms: int = 350
    ):
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

        # Initialize the Soniox-pattern VAD service
        self.vad = VADService(
            sample_rate=output_sample_rate,
            start_threshold=vad_start_threshold,
            end_threshold=vad_end_threshold,
            min_silence_duration_ms=min_silence_duration_ms
        )

    def resample(self, pcm16_bytes: bytes) -> bytes:
        """
        Downsamples PCM16 linear audio to target rate using rational polyphase filtering.
        """
        if not pcm16_bytes:
            return b""

        if self.input_sample_rate == self.output_sample_rate:
            return pcm16_bytes

        audio_int16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if len(audio_int16) == 0:
            return b""

        # 48000 -> 16000 (Downsample factor 1/3)
        resampled = resample_poly(
            audio_int16,
            up=self.output_sample_rate,
            down=self.input_sample_rate
        )

        resampled_int16 = np.clip(resampled, -32768, 32767).astype(np.int16)
        return resampled_int16.tobytes()

    def process_with_vad(self, pcm_bytes: bytes) -> Dict[str, Any]:
        """
        Resamples incoming WebSocket audio and processes it against frame VAD.
        Returns a simplified event contract compatible with your voice router.
        """
        resampled_audio = self.resample(pcm_bytes)
        vad_events = self.vad.process(resampled_audio)

        result: Dict[str, Any] = {
            "speech_started": False,
            "speech_ended": False,
            "is_speech": self.vad.is_speaking,
            "speech_audio": None,
            "duration_ms": 0.0
        }

        for event in vad_events:
            if event.event_type == VADEventType.START_OF_SPEECH:
                result["speech_started"] = True
                result["is_speech"] = True

            elif event.event_type == VADEventType.END_OF_SPEECH:
                result["speech_ended"] = True
                result["is_speech"] = False
                result["speech_audio"] = event.audio_pcm16
                result["duration_ms"] = event.duration_ms

        return result

    def reset(self):
        """Reset internal buffers on new call connection."""
        self.vad.reset()