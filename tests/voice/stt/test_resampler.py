import numpy as np

from src.voice.stt.whisper.resampler import AudioResampler


def test_same_sample_rate():
    resampler = AudioResampler(target_sample_rate=16000)

    audio = np.zeros(1600, dtype=np.int16)

    result = resampler.resample(
        audio,
        source_sample_rate=16000,
    )

    assert len(result) == 1600


def test_8khz_to_16khz():
    resampler = AudioResampler(target_sample_rate=16000)

    audio = np.zeros(8000, dtype=np.int16)

    result = resampler.resample(
        audio,
        source_sample_rate=8000,
    )

    assert len(result) == 16000


def test_empty_audio():
    resampler = AudioResampler(target_sample_rate=16000)

    audio = np.array([], dtype=np.int16)

    result = resampler.resample(
        audio,
        source_sample_rate=8000,
    )

    assert len(result) == 0
