from pathlib import Path

import numpy as np
import soundfile as sf

from src.voice.stt.whisper.service import STTService


def test_stt_service_with_silence(tmp_path):

    audio_path = Path(tmp_path) / "silence.wav"

    audio = np.zeros(
        16000,
        dtype=np.float32,
    )

    sf.write(
        audio_path,
        audio,
        16000,
    )

    stt = STTService(
        model_size="tiny"
    )

    text = stt.transcribe(
        str(audio_path),
        language="en",
    )

    assert isinstance(text, str)
