from src.voice.stt.whisper.audio_decoder import AudioDecoder
from src.voice.stt.whisper.config import STTConfig
from src.voice.stt.whisper.frame_buffer import AudioFrameBuffer
from src.voice.stt.whisper.pipeline import STTPipeline
from src.voice.stt.common.schemas import TranscriptResult
from src.voice.stt.whisper.resampler import AudioResampler
from src.voice.stt.common.provider import STTProvider


class STTAdapter(STTProvider):

    def __init__(
        self,
        sample_rate: int = 16000,
        input_sample_rate: int = 16000,
        encoding: str = "pcm16",
        language: str | None = "en",
    ):
        self.sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        self.encoding = encoding
        self.language = language

        self.config = STTConfig(
            sample_rate=sample_rate,
            language=language,
        )

        self.decoder = AudioDecoder(
            encoding=encoding
        )

        self.resampler = AudioResampler(
            target_sample_rate=sample_rate
        )

        self.frame_buffer = AudioFrameBuffer()

        self.pipeline = STTPipeline(
            config=self.config
        )

    def process_audio(
        self,
        audio_chunk: bytes,
    ) -> list[TranscriptResult]:

        if not audio_chunk:
            return []

        audio = self.decoder.decode(
            audio_chunk
        )

        audio = self.resampler.resample(
            audio,
            self.input_sample_rate,
        )

        pcm_bytes = audio.tobytes()

        frames = self.frame_buffer.add(
            pcm_bytes
        )

        results = []

        for frame in frames:

            result = self.pipeline.process_frame(
                frame
            )

            if result is not None:
                results.append(result)

        return results

    def set_language(
        self,
        language: str | None,
    ):
        self.language = language

        self.pipeline.set_language(
            language
        )

    def reset(self):
        self.frame_buffer.reset()
        self.pipeline.buffer.reset()
