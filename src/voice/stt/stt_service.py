import os
import time
from pathlib import Path
import numpy as np
import transcribe_cpp


# Path relative to this current file
CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(
    CURRENT_DIR
    / "transcribe.cpp"
    / "models"
    / "parakeet-tdt-0.6b-v3"
    / "parakeet-tdt-0.6b-v3-Q8_0.gguf"
)


class STTService:

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        backend: str = "auto",
        n_threads: int = 8,
    ):
        self.model_path = model_path
        self.backend = backend
        self.n_threads = n_threads

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found:\n{self.model_path}"
            )

        print()
        print("================================")
        print("Loading Parakeet STT")
        print("================================")

        print("Model:", self.model_path)
        print("Backend:", self.backend)
        print("Threads:", self.n_threads)

        start = time.perf_counter()

        self.model = transcribe_cpp.Model(
            self.model_path,
            backend=self.backend
        )

        elapsed = time.perf_counter() - start

        print(
            f"Parakeet loaded in {elapsed:.3f}s"
        )

        print(
            "Model remains loaded in memory."
        )

        print("================================")
        print()


    def pcm16_to_float32(
        self,
        pcm_audio: bytes
    ):

        audio = np.frombuffer(
            pcm_audio,
            dtype=np.int16
        )

        audio = audio.astype(
            np.float32
        )

        audio /= 32768.0

        return audio


    def transcribe_pcm16(
        self,
        pcm_audio: bytes
    ) -> str:

        if not pcm_audio:
            return ""

        print()
        print("--------------------------------")
        print("Starting Parakeet")
        print("--------------------------------")

        print(
            "Audio bytes:",
            len(pcm_audio)
        )

        # ================================================
        # PCM16 -> FLOAT32
        # ================================================

        conversion_start = time.perf_counter()

        pcm_float32 = self.pcm16_to_float32(
            pcm_audio
        )

        conversion_time = (
            time.perf_counter()
            - conversion_start
        )

        print(
            f"PCM conversion: "
            f"{conversion_time:.4f}s"
        )

        # ================================================
        # PARakeet
        # ================================================

        start = time.perf_counter()

        result = transcribe_cpp.transcribe(
            self.model,
            pcm_float32,
            backend=self.backend,
            n_threads=self.n_threads,
            language="en",
            timestamps="none"
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        # ================================================
        # RESULT
        # ================================================

        text = ""

        if hasattr(result, "text"):
            text = result.text or ""
        else:
            text = str(result)

        text = text.strip()

        print(
            "Transcript:",
            text
        )

        print(
            f"Inference time: "
            f"{elapsed:.3f}s"
        )

        print("--------------------------------")
        print()

        return text