import os
import numpy as np
import sounddevice as sd

from src.voice.stt.whisper.pipeline import STTPipeline


SAMPLE_RATE = 16000
FRAME_SIZE = 512
DURATION = 15


def main():

    pipeline = STTPipeline(
        sample_rate=SAMPLE_RATE
    )

    print("\n==============================")
    print(" REAL-TIME STT TEST")
    print("==============================")
    print("Speak into your microphone.")
    print("Say something and pause.")
    print("Press Ctrl+C to stop.\n")

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SIZE,
        ) as stream:

            while True:

                audio_frame, overflowed = (
                    stream.read(FRAME_SIZE)
                )

                if overflowed:
                    print(
                        "Warning: audio buffer overflow"
                    )

                audio_frame = (
                    audio_frame[:, 0]
                )

                result = pipeline.process_frame(
                    audio_frame
                )

                if result:

                    print("\n")
                    print("==============================")
                    print("FINAL TRANSCRIPT:")
                    print(result.text)
                    print("==============================")
                    print("\nSpeak again...")

    except KeyboardInterrupt:

        print("\n\nSTT test stopped.")


if __name__ == "__main__":
    main()
