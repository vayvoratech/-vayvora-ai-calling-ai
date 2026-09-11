import numpy as np
from scipy.signal import resample_poly

from src.voice.audio_services.vad_services import VADService


class AudioProcessor:

    def __init__(
        self,
        input_sample_rate=48000,
        output_sample_rate=16000,
        vad_threshold=0.5
    ):

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate

        self.vad = VADService(
            sample_rate=output_sample_rate,
            threshold=vad_threshold
        )

    # ==================================================
    # PCM16 bytes -> NumPy array
    # ==================================================

    def pcm16_to_numpy(
        self,
        pcm_bytes: bytes
    ) -> np.ndarray:

        if not pcm_bytes:

            return np.array(
                [],
                dtype=np.int16
            )

        return np.frombuffer(
            pcm_bytes,
            dtype=np.int16
        )

    # ==================================================
    # Resample audio
    # ==================================================

    def resample_audio(
        self,
        audio: np.ndarray
    ) -> np.ndarray:

        if len(audio) == 0:

            return np.array(
                [],
                dtype=np.int16
            )

        # No conversion required
        if (
            self.input_sample_rate
            ==
            self.output_sample_rate
        ):

            return audio

        # Example:
        # 48000 Hz -> 16000 Hz

        resampled = resample_poly(
            audio,
            self.output_sample_rate,
            self.input_sample_rate
        )

        # Keep valid PCM16 range

        resampled = np.clip(
            resampled,
            -32768,
            32767
        )

        return resampled.astype(
            np.int16
        )

    # ==================================================
    # Process raw browser audio
    # ==================================================

    def process(
        self,
        pcm_bytes: bytes
    ) -> bytes:

        """
        Convert incoming PCM16 audio from the
        browser sample rate to the STT/VAD
        sample rate.

        Input:
            PCM16 48 kHz

        Output:
            PCM16 16 kHz
        """

        audio = self.pcm16_to_numpy(
            pcm_bytes
        )

        if len(audio) == 0:

            return b""

        resampled = self.resample_audio(
            audio
        )

        return resampled.tobytes()

    # ==================================================
    # Process + VAD
    # ==================================================

    def process_with_vad(
        self,
        pcm_bytes: bytes
    ) -> dict:

        """
        Resample incoming audio and run
        stateful VAD.

        Returns:

        {
            speech_started: bool,
            speech_ended: bool,
            is_speech: bool,
            speech_audio: bytes | None
        }
        """

        # ----------------------------------------------
        # 48 kHz -> 16 kHz
        # ----------------------------------------------

        processed_audio = self.process(
            pcm_bytes
        )

        if not processed_audio:

            return {
                "speech_started": False,
                "speech_ended": False,
                "is_speech": False,
                "speech_audio": None
            }

        # ----------------------------------------------
        # Run VAD
        # ----------------------------------------------

        return self.vad.process(
            processed_audio
        )

    # ==================================================
    # Reset
    # ==================================================

    def reset(self):

        self.vad.reset()