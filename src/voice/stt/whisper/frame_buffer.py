import numpy as np


class AudioFrameBuffer:
    """
    Converts arbitrary PCM audio chunks into fixed-size frames.

    Input:
        Any number of int16 PCM samples

    Output:
        Fixed 512-sample frames
    """

    FRAME_SIZE = 512

    def __init__(self):
        self.buffer = np.array(
            [],
            dtype=np.int16,
        )

    def add(
        self,
        audio_chunk: bytes,
    ) -> list[np.ndarray]:
        """
        Add incoming PCM bytes and return
        all complete 512-sample frames.
        """

        if not audio_chunk:
            return []

        incoming = np.frombuffer(
            audio_chunk,
            dtype=np.int16,
        )

        self.buffer = np.concatenate(
            [self.buffer, incoming]
        )

        frames = []

        while len(self.buffer) >= self.FRAME_SIZE:

            frame = self.buffer[
                :self.FRAME_SIZE
            ]

            frames.append(frame)

            self.buffer = self.buffer[
                self.FRAME_SIZE:
            ]

        return frames

    def reset(self):
        """Clear buffered audio."""

        self.buffer = np.array(
            [],
            dtype=np.int16,
        )

    def buffered_samples(self) -> int:
        """Return number of samples waiting."""

        return len(self.buffer)
    