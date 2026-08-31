from datasets import load_dataset, Audio


class STTDataset:
    """
    Loads and prepares an ASR dataset for Wav2Vec2-BERT 2.0.
    """

    def __init__(
        self,
        dataset_name: str,
        language: str = "en",
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.language = language
        self.split = split

    def load(self):
        print(f"Loading dataset: {self.dataset_name}")

        dataset = load_dataset(
            self.dataset_name,
            self.language,
            split=self.split,
        )

        # Resample audio to 16 kHz.
        dataset = dataset.cast_column(
            "audio",
            Audio(sampling_rate=16000),
        )

        print(f"Dataset loaded: {len(dataset)} samples")

        return dataset

    @staticmethod
    def inspect(dataset, num_samples: int = 5):
        print("\nDataset columns:")
        print(dataset.column_names)

        print("\nSample data:")

        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]

            print(f"\nSample {i + 1}")

            print("Audio:")
            print(sample["audio"])

            print("Text:")
            print(sample["text"])


if __name__ == "__main__":

    # Temporary test dataset.
    # We will replace this with our final multilingual
    # training dataset later.

    stt_dataset = STTDataset(
        dataset_name="PolyAI/minds14",
        language="en-US",
        split="train",
    )

    dataset = stt_dataset.load()

    stt_dataset.inspect(dataset)