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

# Language code to full name mapping
LANGUAGE_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "hi": "Hindi",
    "ne": "Nepali",
    "ar": "Arabic",
    "pt": "Portuguese",
    "bn": "Bengali",
    "ur": "Urdu",
    "tr": "Turkish",
    "vi": "Vietnamese"
}

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

# Function to detect language and transcribe audio
def detect_language_and_transcribe(file_path, model_size="medium", device="cpu", compute_type="int8"):
    # Initialize the WhisperModel
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Perform transcription
    segments, info = model.transcribe(file_path, beam_size=5, language=None)  # Auto language detection
    transcribed_text = " ".join([segment.text for segment in segments if hasattr(segment, 'text')])

    # Detect language
    detected_lang_code = info.language
    detected_lang_name = LANGUAGE_MAP.get(detected_lang_code, "Unknown Language")

    # Print detected language and transcription
    print(f"Detected language: '{detected_lang_name}' ({detected_lang_code}) with probability {info.language_probability:.6f}")
    print(f"Transcribed Text: {transcribed_text}")
    return detected_lang_name, transcribed_text

# Function to record and detect language with transcription
def main():
    # Record audio
    audio_file = record_audio(duration=5, file_path="recorded_audio.wav")

    # Detect language and transcribe
    detected_language, transcribed_text = detect_language_and_transcribe(audio_file)
    print(f"Detected Language: {detected_language}")
    print(f"Transcribed Text: {transcribed_text}")

if __name__ == "__main__":
    main()
