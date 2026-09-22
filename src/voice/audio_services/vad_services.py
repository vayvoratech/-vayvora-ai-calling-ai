import collections
from enum import Enum
from typing import Optional, List, NamedTuple
import numpy as np
import torch


class VADEventType(Enum):
    NONE = "none"
    START_OF_SPEECH = "start_of_speech"
    ACTIVE_SPEECH = "active_speech"
    END_OF_SPEECH = "end_of_speech"


class VADEvent(NamedTuple):
    event_type: VADEventType
    audio_pcm16: Optional[bytes] = None
    speech_probability: float = 0.0
    duration_ms: float = 0.0


class VADService:
    def __init__(
        self,
        sample_rate: int = 16000,
        start_threshold: float = 0.30,  # Sensitive enough for standard laptop microphones
        end_threshold: float = 0.15,
        min_speech_duration_ms: int = 64,    # 2 consecutive speech frames
        min_silence_duration_ms: int = 400,  # ~12 silent frames before cutoff
        pre_roll_duration_ms: int = 300,
    ):
        self.sample_rate = sample_rate
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold

        self.frame_samples = 512  # Exactly 32ms at 16kHz
        self.frame_bytes = self.frame_samples * 2

        self.min_speech_frames = max(1, int(min_speech_duration_ms / 32))
        self.min_silence_frames = max(1, int(min_silence_duration_ms / 32))
        self.pre_roll_frames = max(1, int(pre_roll_duration_ms / 32))

        self.pre_roll_buffer = collections.deque(maxlen=self.pre_roll_frames)
        self.is_speaking = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        self.collected_speech_frames: List[bytes] = []

        self.raw_buffer = bytearray()
        self._debug_counter = 0

        print("================================")
        print("Initializing Silero VAD (Soniox Design)")
        print("================================")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True
        )
        self.model.eval()
        self.reset()
        print(f"Sampling Rate: {self.sample_rate}Hz | Frame Size: {self.frame_samples} samples")
        print(f"Thresholds: start={self.start_threshold} | end={self.end_threshold}")
        print("================================\n")

    def reset(self):
        self.raw_buffer.clear()
        self.pre_roll_buffer.clear()
        self.collected_speech_frames.clear()
        self.is_speaking = False
        self.speech_frames_count = 0
        self.silence_frames_count = 0
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def process(self, pcm_bytes: bytes) -> List[VADEvent]:
        events: List[VADEvent] = []
        if not pcm_bytes:
            return events

        self.raw_buffer.extend(pcm_bytes)

        while len(self.raw_buffer) >= self.frame_bytes:
            frame = bytes(self.raw_buffer[:self.frame_bytes])
            del self.raw_buffer[:self.frame_bytes]

            audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            
            # FIX: Ensure 2D shape [batch=1, samples=512] expected by Silero TorchScript
            tensor_chunk = torch.from_numpy(audio_np).unsqueeze(0)

            with torch.no_grad():
                prob = self.model(tensor_chunk, self.sample_rate)
                speech_prob = float(prob.squeeze().item())

            # Periodic diagnostic print to verify audio levels and VAD readings
            self._debug_counter += 1
            if self._debug_counter % 30 == 0:
                max_amp = float(np.max(np.abs(audio_np)))
                print(f"[VAD Meter] Peak Amp: {max_amp:.3f} | Speech Prob: {speech_prob:.3f} | Speaking: {self.is_speaking}")

            if not self.is_speaking:
                self.pre_roll_buffer.append(frame)

                if speech_prob >= self.start_threshold:
                    self.speech_frames_count += 1
                    if self.speech_frames_count >= self.min_speech_frames:
                        self.is_speaking = True
                        self.silence_frames_count = 0
                        self.speech_frames_count = 0

                        # Prepend preserved pre-roll buffer to prevent cutting off words
                        self.collected_speech_frames = list(self.pre_roll_buffer)
                        self.pre_roll_buffer.clear()

                        events.append(VADEvent(
                            event_type=VADEventType.START_OF_SPEECH,
                            speech_probability=speech_prob
                        ))
                else:
                    self.speech_frames_count = 0

            else:
                self.collected_speech_frames.append(frame)

                if speech_prob < self.end_threshold:
                    self.silence_frames_count += 1
                else:
                    self.silence_frames_count = 0

                if self.silence_frames_count >= self.min_silence_frames:
                    prune_count = self.min_silence_frames
                    speech_frames = (
                        self.collected_speech_frames[:-prune_count]
                        if len(self.collected_speech_frames) > prune_count
                        else self.collected_speech_frames
                    )

                    complete_audio = b"".join(speech_frames)
                    duration_ms = (len(complete_audio) / 2 / self.sample_rate) * 1000

                    events.append(VADEvent(
                        event_type=VADEventType.END_OF_SPEECH,
                        audio_pcm16=complete_audio,
                        speech_probability=speech_prob,
                        duration_ms=duration_ms
                    ))

                    self.is_speaking = False
                    self.silence_frames_count = 0
                    self.collected_speech_frames.clear()
                    if hasattr(self.model, "reset_states"):
                        self.model.reset_states()
                else:
                    events.append(VADEvent(
                        event_type=VADEventType.ACTIVE_SPEECH,
                        speech_probability=speech_prob
                    ))

        return events