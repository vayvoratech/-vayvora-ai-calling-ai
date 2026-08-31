import torch
from transformers import Wav2Vec2BertForCTC


MODEL_NAME = "facebook/w2v-bert-2.0"
TOKENIZER_PATH = "models/stt/tokenizer"


class STTModel:
    """
    Wav2Vec2-BERT 2.0 configured for CTC-based ASR fine-tuning.
    """

    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Loading model on: {self.device}")

        self.model = Wav2Vec2BertForCTC.from_pretrained(
            MODEL_NAME,

            # ASR configuration
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            mask_time_prob=0.0,
            layerdrop=0.0,

            ctc_loss_reduction="mean",

            # Lightweight adapter for fine-tuning
            add_adapter=True,

            # Our vocabulary
            pad_token_id=0,

            vocab_size=self._get_vocab_size(),
        )

        self.model.to(self.device)

        print("Wav2Vec2-BERT model loaded successfully.")

    @staticmethod
    def _get_vocab_size():
        import json

        with open(
            f"{TOKENIZER_PATH}/vocab.json",
            "r",
            encoding="utf-8",
        ) as file:
            vocabulary = json.load(file)

        return len(vocabulary)

    def get_model(self):
        return self.model


if __name__ == "__main__":

    stt_model = STTModel()

    model = stt_model.get_model()

    print("\nModel configuration:")
    print(f"Device: {stt_model.device}")
    print(f"Vocabulary size: {model.config.vocab_size}")