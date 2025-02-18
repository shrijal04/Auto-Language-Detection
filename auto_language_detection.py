# pip install faster_whisper

import pyaudio
import numpy as np
import wave
from faster_whisper import WhisperModel

# Define constants for recording
SAMPLING_RATE = 16000  # 16 kHz sampling rate
CHANNELS = 1  # Mono audio
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHUNK_SIZE = 1024  # Number of frames per buffer
RECORD_DURATION = 5  # Duration in seconds

# Function to record audio for a specific duration
def record_audio(duration=RECORD_DURATION, file_path="recorded_audio.wav"):
    p = pyaudio.PyAudio()

    # Start the stream for recording
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLING_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    print("Recording...")

    frames = []
    for _ in range(0, int(SAMPLING_RATE / CHUNK_SIZE * duration)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Write recorded data to a .wav file
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLING_RATE)
        wf.writeframes(b''.join(frames))

    print(f"Recording saved to {file_path}")
    return file_path

# Function to detect language without full transcription
def detect_language(file_path, model_size="medium", device="cpu", compute_type="int8"):
    # Initialize the WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Perform language detection
    _, info = model.transcribe(file_path, beam_size=5, language=None)

    # Print detected language
    print(f"Detected language: '{info.language}' with probability {info.language_probability:.6f}")
    return info.language

# Function to record and detect language
def main():
    # Record audio
    audio_file = record_audio(duration=5, file_path="recorded_audio.wav")

    # Detect language
    detected_language = detect_language(audio_file)
    print(f"Detected Language: {detected_language}")

if __name__ == "__main__":
    main()
