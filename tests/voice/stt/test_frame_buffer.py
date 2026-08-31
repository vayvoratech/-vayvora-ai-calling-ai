import numpy as np

from src.voice.stt.whisper.frame_buffer import AudioFrameBuffer


def test_exact_frame():
    buffer = AudioFrameBuffer()

    audio = np.zeros(512, dtype=np.int16).tobytes()

    frames = buffer.add(audio)

    assert len(frames) == 1
    assert len(frames[0]) == 512


def test_partial_frame():
    buffer = AudioFrameBuffer()

    audio = np.zeros(320, dtype=np.int16).tobytes()

    frames = buffer.add(audio)

    assert len(frames) == 0
    assert buffer.buffered_samples() == 320


def test_multiple_frames():
    buffer = AudioFrameBuffer()

    audio = np.zeros(1024, dtype=np.int16).tobytes()

    frames = buffer.add(audio)

    assert len(frames) == 2
    assert len(frames[0]) == 512
    assert len(frames[1]) == 512


def test_reset():
    buffer = AudioFrameBuffer()

    audio = np.zeros(320, dtype=np.int16).tobytes()

    buffer.add(audio)
    buffer.reset()

    assert buffer.buffered_samples() == 0
