import time

import numpy as np
import torch

from src.voice.stt.whisper.audio_buffer import AudioBuffer
from src.voice.stt.whisper.config import STTConfig
from src.voice.stt.common.schemas import TranscriptResult
from src.voice.stt.whisper.service import STTService
from src.voice.stt.whisper.vad import VoiceActivityDetector


class STTPipeline:

    def __init__(
        self,
        config: STTConfig | None = None,
    ):
        self.config = config or STTConfig()

        print("Initializing STT pipeline...")

        self.vad = VoiceActivityDetector(
            threshold=self.config.vad_threshold
        )

        self.buffer = AudioBuffer(
            sample_rate=self.config.sample_rate,
            silence_duration=self.config.silence_duration,
        )

        self.stt = STTService(
            model_size=self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

        # Streaming state
        self.speech_started = False
        self.last_partial_time = 0.0
        self.last_partial_text = ""

        print("STT pipeline ready.")

    def process_frame(
        self,
        audio_frame: np.ndarray,
    ) -> TranscriptResult | None:

        if len(audio_frame) != 512:
            raise ValueError(
                f"Expected 512 samples, "
                f"received {len(audio_frame)}"
            )

        audio_tensor = torch.from_numpy(
            audio_frame
        ).float()

        is_speech = self.vad.is_speech(
            audio_tensor,
            self.config.sample_rate,
        )

        if is_speech:
            self.speech_started = True

        utterance = self.buffer.add_frame(
            audio_frame,
            is_speech,
        )

        # ------------------------------------------------
        # FINAL TRANSCRIPT
        # ------------------------------------------------

        if utterance is not None:

            self.speech_started = False
            self.last_partial_time = 0.0
            self.last_partial_text = ""

            return self._transcribe(
                utterance,
                is_final=True,
            )

        # ------------------------------------------------
        # PARTIAL TRANSCRIPT
        # ------------------------------------------------

        if (
            self.config.partial_enabled
            and self.speech_started
        ):

            now = time.monotonic()

            elapsed = (
                now - self.last_partial_time
            )

            audio_duration = (
                self.buffer.buffered_samples()
                / self.config.sample_rate
            )

            if (
                audio_duration
                >= self.config.minimum_partial_duration
                and (
                    self.last_partial_time == 0
                    or elapsed
                    >= self.config.partial_interval
                )
            ):

                audio = self.buffer.get_audio()

                if audio is not None:

                    result = self._transcribe(
                        audio,
                        is_final=False,
                    )

                    if result is not None:

                        if (
                            result.text
                            and result.text
                            != self.last_partial_text
                        ):

                            self.last_partial_text = (
                                result.text
                            )

                            self.last_partial_time = (
                                now
                            )

                            return result

        return None

    def _transcribe(
        self,
        audio: np.ndarray,
        is_final: bool,
    ) -> TranscriptResult | None:

        if audio is None or len(audio) == 0:
            return None

        audio_float = (
            audio.astype(np.float32)
            / 32768.0
        )

        result = self.stt.transcribe_audio(
            audio_float,
            sample_rate=self.config.sample_rate,
            language=self.config.language,
        )

        text = result["text"].strip()

        if not text:
            return None

        return TranscriptResult(
            text=text,
            is_final=is_final,
            language=result["language"],
        )

    def set_language(
        self,
        language: str | None,
    ):
        self.config.language = language

    def reset(self):

        self.buffer.reset()

        self.speech_started = False
        self.last_partial_time = 0.0
        self.last_partial_text = ""
