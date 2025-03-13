import webrtcvad
import pyaudio
import wave
import io
from openai import OpenAI
import numpy as np

class VerbalInteraction:
    def __init__(self, enable_speech: bool = False, enable_listening: bool = False, enable_reasoning: bool = False, think_out_loud: bool = False):
        self.speech = enable_speech
        self.listening = enable_listening
        self.reasoning = enable_reasoning
        self.think_out_loud = think_out_loud
        self.message_history = []  # to store conversation history

        try:
            with open('utils/secret/openai_api_key', 'r') as f:
                self.api_key = f.read().strip()
                self.client = OpenAI(api_key=self.api_key)
        except FileNotFoundError:
            print("Can't initialize OpenAI API without a key. Please create a file named 'utils/secret/openai_api_key' with your OpenAI API key.")
            self.speech = False 
            self.listening = False
            self.reasoning = False

        self.pyaudio_instance = pyaudio.PyAudio()
        
        if self.speech:
            self.out_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True
            )
        
        if self.listening:
            # Set frames_per_buffer to match our chosen frame duration (see below)
            self.in_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=32000,
                input=True,
                frames_per_buffer=960  # 20ms at 24000 Hz (can adjust as needed)
            )
            # Initialize the WebRTC VAD with aggressiveness mode 2 (balanced sensitivity)
            self.vad = webrtcvad.Vad(2)

    def speak(self, text: str):
        print("SPEAK:", text)
        if self.speech:
            # Use text-to-speech API
            with self.client.audio.with_streaming_response.speech.create(
                model="tts-1",
                voice="shimmer",
                input=text,
                response_format="pcm"
            ) as response:
                for chunk in response.iter_bytes(1024):
                    self.out_stream.write(chunk)

    def listen(self) -> str:
        """
        Listen to audio input using py-webrtcvad for VAD.
        Records in small chunks (30ms frames) until it detects speech and then stops
        after a set number of consecutive non-speech frames.
        """
        if not self.listening:
            return input("LISTEN: ")

        print("Listening for speech...")

        sample_rate = 32000
        frame_duration_ms = 30  # Duration of each frame in milliseconds (10, 20, or 30 ms are supported)
        frame_size = int(sample_rate / 1000.0 * frame_duration_ms)  # number of samples per frame
        frames = []
        silence_threshold = 50  # number of consecutive silent frames to consider as the end of speech
        silence_counter = 0
        start_counter = 0
        speech_started = False

        # Loop until we detect that speech has begun
        while not speech_started:
            frame = self.in_stream.read(frame_size)
            if self.vad.is_speech(frame, sample_rate):
                start_counter += 1
                frames.append(frame)
            else:
                start_counter = 0

            if start_counter > 5:
                speech_started = True
                frames.append(frame)
                print("Speech detected. Recording...")
            # Else: continue waiting without storing noise

        # Continue recording until we hit the silence threshold
        while True:
            frame = self.in_stream.read(frame_size)
            is_speech = self.vad.is_speech(frame, sample_rate)
            frames.append(frame)
            if is_speech:
                silence_counter = 0
            else:
                silence_counter += 1

            if silence_counter > silence_threshold:
                print("Silence detected, ending recording.")
                break

        # Write recorded frames into an in-memory WAV file
        audio_buffer = io.BytesIO()
        wf = wave.open(audio_buffer, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        audio_buffer.seek(0)

        with open("recorded_audio.wav", "wb") as f:
            f.write(audio_buffer.read())

        audio_file = open("recorded_audio.wav", "rb")

        # Use OpenAI Whisper to transcribe the audio
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        print("HEARD:", transcription)
        return transcription

    def think(self, prompt: str):
        if self.think_out_loud:
            self.speak(prompt)
        else:
            print("THINKING:", prompt)

    def ask(self, prompt: str):
        self.speak(prompt)
        return self.listen()

    def reason(self, prompt: str):
        if not self.reasoning:
            print("Reasoning is disabled.")
            return

        self.message_history.append({"role": "user", "content": prompt})
        self.message_history = self.message_history[-10:]
        
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.message_history
        )
        assistant_response = completion.choices[0].message.content
        self.message_history.append({"role": "assistant", "content": assistant_response})
        self.speak(assistant_response)


if __name__ == "__main__":
    vi = VerbalInteraction(enable_speech=False, enable_listening=False, enable_reasoning=True)
    
    # Record input using improved VAD-based listening
    user_input = vi.listen()

    # Process the user input via reasoning
    vi.reason(user_input)
