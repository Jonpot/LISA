from typing import List, Tuple
import webrtcvad
import pyaudio
import wave
import io
from openai import OpenAI
import numpy as np
import tkinter as tk
from tkinter import filedialog
import PyPDF2

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

    def speak(self, text: str) -> None:
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

    def think(self, prompt: str) -> None:
        if self.think_out_loud:
            self.speak(prompt)
        else:
            print("THINKING:", prompt)

    def ask(self, prompt: str) -> str:
        self.speak(prompt)
        return self.listen()

    def ask_boolean(self, prompt: str) -> bool:
        response = self.ask(prompt).lower()
        if response == "y": # simple typing of 'y' or 'n' for yes or no
            return True
        elif response == "n":
            return False
        
        # otherwise, likely processing spoken response
        parsed_words = response.split()
        affirmative_words = ["yes", "true", "1", "yeah", "yep", "yup", "sure", "ok", "okay", "ready", "continue"]
        negative_words = ["no", "false", "0", "nope", "nah", "nay", "not", "never", "cancel"]
        
        # if any of the negative words are in the response, return False
        # do this first for safety, in case the response contains both affirmative and negative words
        for word in parsed_words:
            if word.strip().lower() in negative_words:
                self.think(f"I heard '{word}' in the response, so I'll assume you meant 'no'.")
                return False
            
        # if any of the affirmative words are in the response, return True
        for word in parsed_words:
            if word.strip().lower() in affirmative_words:
                self.think(f"I heard '{word}' in the response, so I'll assume you meant 'yes'.")
                return True
        
        # if the response is not clear, ask again
        if self.reasoning:
            reasoning_response = self.reason(f"The user said '{response}'. Do you think this is an 'affirmative' or 'negative' or 'unclear' answer? Reply with only one of these words.")
            if reasoning_response == "affirmative":
                return True
            elif reasoning_response == "negative":
                return False

        return self.ask_boolean(prompt)

    def ask_position_type(self, prompt: str) -> str:
        response = self.ask(prompt).lower()

        if response == "object_dock":
            return "object_dock"
        elif response == "swap":
            return "swap"
        elif response == "storage":
            return "storage"
        
        # otherwise, likely processing spoken response or some other type of storage
        parsed_words = response.split()
        object_dock_words = ["bench", "dock", "desk", "me", "workstation", "human", "offloading"]
        swap_words = ["swap", "exchange", "trade", "switch", "empty"]

        if any(word.strip().lower() in object_dock_words for word in parsed_words):
            word = next((word for word in parsed_words if word.strip().lower() in object_dock_words), None)
            self.think(f"I heard '{word}' in the response, so I'll assume you meant 'object_dock'.")
            return "object_dock"
        elif any(word.strip().lower() in swap_words for word in parsed_words):
            word = next((word for word in parsed_words if word.strip().lower() in swap_words), None)
            self.think(f"I heard '{word}' in the response, so I'll assume you meant 'swap'.")
            return "swap"
        
        if self.reasoning:
            reasoning_response = self.reason(f"The user said '{response}'. Do you think this is an 'object_dock', 'swap', or 'storage'? Reply with only one of these words. For context: an object_dock is a place like a workbench or manned station where a humans alone will be interacting with objects brought here (e.g. 'bench, me, human, desk'), a swap is a place to exchange an object that is always intentionally empty and will only interact with a robot (e.g. 'swap, exchange, trade'), and storage is anything that doesn't obviously fit into either of those two categories.")
            if reasoning_response == "object_dock":
                return "object_dock"
            elif reasoning_response == "swap":
                return "swap"

        # If it's not an object_dock or swap, assume it's exactly whatever they said ("storage" doesn't have any special functionality, it's
        # just "not a dock or swap")
        return response


    def ask_sds(self, existing_objects: List[str], storage_types: List[str]) -> Tuple[str, bool, str, str]:
        """
        Opens a prompt to select/drag-and-drop a PDF file of a SDS (Safety Data Sheet) for a chemical/reagent.
        The SDS is then uploaded to the reasoning LLM (OpenAI's chatgpt in this case) to extract the relevant information:
        - Name of this object (must be distinct from existing objects)
        - Does this require "special handling" (should the robot move much slower when moving this object)
        - What type of storage should be used for this object (out of the given storage types)
        - A brief description of this object (for the robot to use as a reference when asked about it later)
        """
        # Open a GUI to select a PDF file of the SDS
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select a PDF file of the SDS", filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            print("No file selected.")
            return None, False, None, None
        
        # Now, we need to extract the text from the PDF file  
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        # Now, ask the reasoning LLM to extract the relevant information from the text
        # Tell it to return the answers in a specific JSON format
        object_name = None 
        special_handling = None
        storage_type = None
        description = None
        messages = [
                    {"role": "user", "content": f"Please extract the following information from this SDS text:\n\n{text}\n\n1. Name of this object (must be distinct from existing objects: {existing_objects}) as a str\n2. Does this require special handling ('true' or 'false')?\n3. What type of storage should be used for this object (out of the given storage types: {storage_types})? as a str\n4. A brief description of this object.\n\nPlease return the answers in the following JSON format:\n{'{'}object_name': '...' <str>, 'special_handling': '...' <bool>, 'storage_type': '...' <str>, 'description': '...' <str>{'}'}. Do not include any other text or explanation. Just return the JSON object."}
                ]
        attempts = 0
        while object_name is None or special_handling is None or storage_type is None:
            attempts += 1
            if attempts > 3:
                self.think("Too many attempts to get valid SDS information. Moving to manual input.")
                return None, False, None, None
            response = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages
            )
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            try:
                response_json = eval(response.choices[0].message.content)
                object_name = response_json.get('object_name')
                special_handling = response_json.get('special_handling')
                storage_type = response_json.get('storage_type')
                description = response_json.get('description')
            except Exception as e:
                self.think(f"Error parsing response: {e}")
                self.think("Response:", response.choices[0].message.content)
                messages.append({"role": "user", "content": "I couldn't parse the response as JSON. Please try again. Do not include any other text or explanation."})

                object_name = None
                special_handling = None
                storage_type = None
                description = None
                continue

            # Check if the object name is distinct from existing objects
            if object_name in existing_objects:
                messages.append({"role": "user", "content": f"The object name '{object_name}' is already in use. Please provide a distinct name. Return the entire JSON response with the new name and no other text."})
                object_name = None
                continue

            # Check if the special handling is a boolean
            if special_handling not in ["true", "false"]:
                messages.append({"role": "user", "content": f"The special handling value '{special_handling}' is not valid. Please provide 'true' or 'false'. Return the entire JSON response with the new value and no other text."})
                special_handling = None
                continue

            # Check if the storage type is valid
            if storage_type not in storage_types:
                messages.append({"role": "user", "content": f"The storage type '{storage_type}' is not valid. Please provide a valid storage type from the list: {storage_types}. Return the entire JSON response with the new value and no other text."})
                storage_type = None
                continue

            # Otherwise, we're done!
            self.think(f"Got the following information from the SDS:\nObject name: {object_name}\nSpecial handling: {special_handling}\nStorage type: {storage_type}\nDescription: {description}")
        return object_name, special_handling == "true", storage_type, description

    def parse_protocol(self, existing_objects: List[str]) -> List[str]:
        """
        Allows a scientist to upload a protocol in PDF, text, etc. format.
        The protocol is then uploaded to the reasoning LLM (OpenAI's chatgpt in this case) to extract the relevant information:
        - What objects (from the existing objects) are needed for this protocol?
        """
        # Open a GUI to select a PDF file of the protocol
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select a PDF file of the protocol", filetypes=[("PDF files", "*.pdf"), ("Text files", "*.txt")])
        if not file_path:
            print("No file selected.")
            return []

        # Now, we need to extract the text from the PDF file  
        # If it's a PDF, use PyPDF2 to extract text
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
        # If it's a text file, just read it
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                text = f.read()
        else:
            print("Unsupported file type. Please select a PDF or text file.")
            return []

        # Now, ask the reasoning LLM to extract the relevant information from the text
        # Tell it to return the answers in a specific JSON format
        object_names = []
        messages = [
                    {"role": "user", "content": f"Please extract the following information from this protocol text:\n\n{text}\n\nWhat objects (from the existing objects: {existing_objects}) are needed for this protocol? Please return the answers as a list of strings. Do not include any other text or explanation. Just return the list."}
                ]
        attempts = 0
        while len(object_names) == 0:
            attempts += 1
            if attempts > 3:
                self.think("Too many attempts to get valid information. Moving to manual input.")
                return []
            response = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=messages
            )
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            try:
                object_names = eval(response.choices[0].message.content)
            except Exception as e:
                self.think(f"Error parsing response: {e}")
                self.think("Response:", response.choices[0].message.content)
                messages.append({"role": "user", "content": "I couldn't parse the response as JSON. Please try again. Do not include any other text or explanation."})
                object_names = []
                continue

            # Check if the objects are in the existing objects
            for object_name in object_names:
                if object_name not in existing_objects:
                    messages.append({"role": "user", "content": f"The object name '{object_name}' is not in the existing objects. Please provide a valid name from the list: {existing_objects}. Return the entire JSON response with the new value and no other text."})
                    object_names = []
                    continue

            # Otherwise, we're done!
            self.think(f"Got the following objects from the protocol:\n{object_names}")
        return object_names

    def respond(self, prompt: str) -> str:
        response = self.reason(prompt)
        if response:
            self.speak(response)
        return response

    def reason(self, prompt: str) -> str | None:
        if not self.reasoning:
            print("Reasoning is disabled.")
            return None

        self.message_history.append({"role": "user", "content": prompt})
        self.message_history = self.message_history[-10:]
        
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=self.message_history
        )
        assistant_response = completion.choices[0].message.content
        self.message_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response


if __name__ == "__main__":
    vi = VerbalInteraction(enable_speech=False, enable_listening=False, enable_reasoning=True)
    
    # Record input using improved VAD-based listening
    user_input = vi.listen()

    # Process the user input via reasoning
    vi.respond(user_input)
