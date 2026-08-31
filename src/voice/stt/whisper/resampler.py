import numpy as np
from scipy.signal import resample_poly


class AudioResampler:
    """
    Resamples audio to the STT model's target rate.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
    ):
        self.target_sample_rate = target_sample_rate

    def resample(
        self,
        audio: np.ndarray,
        source_sample_rate: int,
    ) -> np.ndarray:

        if audio.size == 0:
            return np.array(
                [],
                dtype=np.int16,
            )

        if source_sample_rate == self.target_sample_rate:
            return audio

        gcd = np.gcd(
            source_sample_rate,
            self.target_sample_rate,
        )

        up = self.target_sample_rate // gcd
        down = source_sample_rate // gcd

        audio_float = audio.astype(
            np.float32
        )

        resampled = resample_poly(
            audio_float,
            up,
            down,
        )

        resampled = np.clip(
            resampled,
            -32768,
            32767,
        )

        return resampled.astype(
            np.int16
        )