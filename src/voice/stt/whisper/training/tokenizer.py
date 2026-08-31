import json
import re
from pathlib import Path

from transformers import Wav2Vec2CTCTokenizer


class STTTokenizerBuilder:

    def __init__(self, output_dir="models/stt/tokenizer"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().strip()

        # Keep English, Hindi, Telugu, numbers and basic characters
        text = re.sub(
            r"[^a-zA-Z0-9\u0900-\u097F\u0C00-\u0C7F\s]",
            "",
            text,
        )

        text = re.sub(r"\s+", " ", text)

        return text

    def build_vocabulary(self, transcripts):

        normalized = [
            self.normalize_text(text)
            for text in transcripts
        ]

        all_text = " ".join(normalized)

        vocabulary = sorted(set(all_text))

        vocab_dict = {}

        for token in vocabulary:

            if token == " ":
                vocab_dict["|"] = len(vocab_dict)
            else:
                vocab_dict[token] = len(vocab_dict)

        # CTC special tokens
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        vocab_path = self.output_dir / "vocab.json"

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(
                vocab_dict,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Vocabulary saved: {vocab_path}")
        print(f"Vocabulary size: {len(vocab_dict)}")

        return vocab_dict

    def create_tokenizer(self):

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            str(self.output_dir),
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )

        tokenizer.save_pretrained(
            str(self.output_dir)
        )

        print("CTC tokenizer created successfully.")

        return tokenizer


if __name__ == "__main__":

    sample_transcripts = [
        "hello how are you",
        "i want to book an appointment",
        "नमस्ते मुझे अपॉइंटमेंट चाहिए",
        "मुझे डॉक्टर से बात करनी है",
        "నాకు అపాయింట్మెంట్ కావాలి",
        "నాకు డాక్టర్ తో మాట్లాడాలి",
    ]

    builder = STTTokenizerBuilder()

    builder.build_vocabulary(sample_transcripts)

    tokenizer = builder.create_tokenizer()

    print("\nTokenizer ready.")