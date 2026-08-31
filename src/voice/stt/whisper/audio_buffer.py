import numpy as np


class AudioBuffer:

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_duration: float = 0.8,
    ):
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration

        self.frame_size = 512

        self.silence_frames_required = max(
            1,
            int(
                silence_duration
                * sample_rate
                / self.frame_size
            ),
        )

        self.speech_buffer: list[np.ndarray] = []

        self.silence_frames = 0
        self.has_speech = False

    def add_frame(
        self,
        audio_frame: np.ndarray,
        is_speech: bool,
    ) -> np.ndarray | None:

        if is_speech:

            self.speech_buffer.append(
                audio_frame.copy()
            )

            self.silence_frames = 0
            self.has_speech = True

            return None

        if not self.has_speech:
            return None

        self.silence_frames += 1

        if (
            self.silence_frames
            < self.silence_frames_required
        ):
            return None

        utterance = self.get_audio()

        self.reset()

        return utterance

    def get_audio(self) -> np.ndarray | None:

        if not self.speech_buffer:
            return None

        return np.concatenate(
            self.speech_buffer
        )

    def buffered_samples(self) -> int:

        if not self.speech_buffer:
            return 0

        return sum(
            len(frame)
            for frame in self.speech_buffer
        )

    def buffered_duration(self) -> float:

        return (
            self.buffered_samples()
            / self.sample_rate
        )

    def reset(self):

        self.speech_buffer.clear()

        self.silence_frames = 0
        self.has_speech = False