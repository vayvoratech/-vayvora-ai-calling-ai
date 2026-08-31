import numpy as np
import sounddevice as sd

from src.voice.stt.whisper.adapter import STTAdapter


SAMPLE_RATE = 16000
CHUNK_SIZE = 320


def main():

    print("Initializing STT adapter...")

    stt = STTAdapter(
        sample_rate=SAMPLE_RATE
    )

    print("\n================================")
    print(" WEBSOCKET STT ADAPTER TEST")
    print("================================")
    print("Speak normally.")
    print("Pause after each sentence.")
    print("Press Ctrl+C to stop.\n")

    audio_queue = []

    def audio_callback(
        indata,
        frames,
        time_info,
        status,
    ):
        if status:
            print(
                f"Audio status: {status}"
            )

        audio_queue.append(
            indata[:, 0].copy()
        )

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):

            while True:

                if not audio_queue:
                    continue

                audio = audio_queue.pop(0)

                results = stt.process_audio(
                    audio.tobytes()
                )

                for result in results:

                    print("\n")
                    print("--------------------------------")
                    print("FINAL TRANSCRIPT")
                    print("--------------------------------")
                    print(result.text)
                    print(
                        f"Language: {result.language}"
                    )
                    print("--------------------------------")

    except KeyboardInterrupt:

        print("\nSTT adapter test stopped.")


if __name__ == "__main__":
    main()
