import audioop
import numpy as np


class AudioDecoder:
    """
    Decode telephony audio into PCM16.

    Supported:
        - pcm16
        - mulaw / G.711 μ-law
    """

    def __init__(self, encoding: str = "pcm16"):
        self.encoding = encoding.lower()

    def decode(self, audio_bytes: bytes) -> np.ndarray:

        if not audio_bytes:
            return np.array([], dtype=np.int16)

        if self.encoding == "pcm16":
            return np.frombuffer(
                audio_bytes,
                dtype=np.int16,
            ).copy()

        if self.encoding in {
            "mulaw",
            "ulaw",
            "g711_ulaw",
        }:
            pcm_bytes = audioop.ulaw2lin(
                audio_bytes,
                2,
            )

            return np.frombuffer(
                pcm_bytes,
                dtype=np.int16,
            ).copy()

        raise ValueError(
            f"Unsupported audio encoding: "
            f"{self.encoding}"
        )