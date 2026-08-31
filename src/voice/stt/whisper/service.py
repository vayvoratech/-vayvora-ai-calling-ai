from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


class STTService:
    """
    Memory-efficient Whisper STT service.

    Designed for CPU-only development machines.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        print(
            f"Loading Whisper model: {model_size}"
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=2,
            num_workers=1,
        )

        print("STT model loaded.")

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = "en",
    ) -> dict:

        if audio is None or len(audio) == 0:
            return {
                "text": "",
                "language": language,
            }

        # Keep audio in float32.
        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
        )

        text_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                text_parts.append(text)

        text = " ".join(text_parts).strip()

        return {
            "text": text,
            "language": info.language,
        }

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
    ) -> str:

        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
        )

        text_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                text_parts.append(text)

        return " ".join(text_parts).strip()