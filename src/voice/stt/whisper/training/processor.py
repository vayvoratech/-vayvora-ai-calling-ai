from transformers import (
    SeamlessM4TFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2BertProcessor,
)


MODEL_NAME = "facebook/w2v-bert-2.0"
TOKENIZER_PATH = "models/stt/tokenizer"


class STTProcessor:

    def __init__(self):

        print("Loading feature extractor...")

        self.feature_extractor = (
            SeamlessM4TFeatureExtractor.from_pretrained(
                MODEL_NAME
            )
        )

        print("Loading tokenizer...")

        self.tokenizer = (
            Wav2Vec2CTCTokenizer.from_pretrained(
                TOKENIZER_PATH,
                unk_token="[UNK]",
                pad_token="[PAD]",
                word_delimiter_token="|",
            )
        )

        print("Creating Wav2Vec2-BERT processor...")

        self.processor = Wav2Vec2BertProcessor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
        )

        print("Processor created successfully.")

    def prepare_sample(self, sample):

        audio = sample["audio"]

        inputs = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        )

        labels = self.processor(
            text=sample["transcription"]
        ).input_ids

        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
        }


if __name__ == "__main__":

    processor = STTProcessor()

    print("\nSTT processor is ready.")