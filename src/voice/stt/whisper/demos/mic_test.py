import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5


print("Available audio devices:\n")
print(sd.query_devices())

print("\nDefault input device:")
print(sd.default.device)

print("\n🎤 Speak for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
)

sd.wait()

print("\nRecording completed.")

print("Maximum amplitude:", audio.max())
print("Minimum amplitude:", audio.min())
print("Average amplitude:", abs(audio).mean())