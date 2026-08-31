import torch
from silero_vad import load_silero_vad


class VoiceActivityDetector:

    def __init__(self, threshold: float = 0.25):

        self.threshold = threshold

        print("Loading Silero VAD...")

        self.model = load_silero_vad()

        print("VAD loaded successfully.")

    def is_speech(
        self,
        audio_frame,
        sample_rate: int = 16000,
    ) -> bool:

        audio = torch.as_tensor(
            audio_frame,
            dtype=torch.float32,
        )

        # Convert int16 PCM → float32
        if audio.abs().max() > 1:
            audio = audio / 32768.0

        if audio.ndim > 1:
            audio = audio.squeeze()

        expected_samples = (
            512
            if sample_rate == 16000
            else 256
        )

        if len(audio) != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples, "
                f"received {len(audio)}"
            )

        with torch.no_grad():

            probability = self.model(
                audio,
                sample_rate,
            ).item()

        return probability >= self.threshold

    def probability(
        self,
        audio_frame,
        sample_rate: int = 16000,
    ) -> float:

        audio = torch.as_tensor(
            audio_frame,
            dtype=torch.float32,
        )

        if audio.abs().max() > 1:
            audio = audio / 32768.0

        if audio.ndim > 1:
            audio = audio.squeeze()

        expected_samples = (
            512
            if sample_rate == 16000
            else 256
        )

        if len(audio) != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples, "
                f"received {len(audio)}"
            )

        with torch.no_grad():

            return self.model(
                audio,
                sample_rate,
            ).item()