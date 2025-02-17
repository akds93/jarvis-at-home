#!/usr/bin/env python3
import json
import requests
import subprocess
import speech_recognition as sr
import pyttsx3
import platform
import time
import os

try:
    import distro
except ImportError:
    distro = None

from inputimeout import inputimeout, TimeoutOccurred

# Global constants for Ollama API and model names
OLLAMA_URL = "http://localhost:11434/api/generate"
CONVERSATIONAL_MODEL = "jarvis-at-home-model_llama3.2:3Bv1"        
COMMAND_MODEL = "jarvis-at-home-commands_model_qwen2.5-coder:3Bv3"   

# --- System Information ---
def get_system_info():
    os_name = platform.system()
    if os_name == "Linux":
        # Try to get distro information; fall back if not available
        if distro:
            dist_name = distro.name() or "Linux"
            dist_version = distro.version() or ""
        else:
            dist_name = "Linux"
            dist_version = ""
        # Get desktop environment info from environment variables
        de = os.environ.get("XDG_CURRENT_DESKTOP") or os.environ.get("DESKTOP_SESSION") or "Unknown DE"
        return f"{os_name} ({dist_name} {dist_version}, {de})"
    else:
        return os_name

SYSTEM_INFO = get_system_info()

# --- TTS Initialization ---
tts_engine = pyttsx3.init()
tts_engine.setProperty('voice', 26)  # Adjust index as needed for a natural-sounding voice
tts_engine.setProperty('rate', 170)  # Lower speaking rate for clarity

def text_to_speech(text):
    """Convert text to speech using pyttsx3."""
    tts_engine.say(text)
    tts_engine.runAndWait()

# --- Helper to clean JSON responses ---
def clean_json_response(text):
    """Remove markdown code block markers from the text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

# --- Voice Confirmation with Fallback ---
def voice_confirmation(prompt_text, timeout=5):
    """
    Ask for confirmation by speaking the prompt and listening for the response.
    Returns True if the response contains 'yes' or 'run it', otherwise False.
    Falls back to typed input if no voice is detected.
    """
    text_to_speech(prompt_text)
    print(prompt_text)
    try:
        response = listen_audio(timeout=timeout)
        if response:
            print("DEBUG: Voice confirmation response:", response)
            if "yes" in response.lower() or "run it" in response.lower():
                return True
            else:
                return False
        else:
            fallback = input("No voice input detected. Type yes or no: ").strip().lower()
            return fallback == "yes"
    except Exception as e:
        print("Voice confirmation error:", e)
        return False

# --- Audio Capture and Speech-to-Text ---
def listen_audio(timeout=15):
    """Listen for audio from the microphone and return the transcribed text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"Listening (up to {timeout} seconds)...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text
        except sr.WaitTimeoutError:
            print(f"No speech detected within {timeout} seconds.")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        except sr.RequestError as e:
            print("STT error:", e)
            return None

# --- Ollama API Interaction ---
def query_ollama(model, prompt, stream=False):
    """
    Send a prompt to the Ollama API using the specified model.
    Returns the JSON response or None if there's an error.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.ok:
            return response.json()
        else:
            print("Ollama API error:", response.text)
            return None
    except Exception as e:
        print("Error querying Ollama:", e)
        return None

# --- Command Detection and Processing ---
def detect_command(text):
    """Simple keyword-based command detection with debug output."""
    command_keywords = ["open", "launch", "execute", "run", "shutdown"]
    text_lower = text.lower()
    for keyword in command_keywords:
        if keyword in text_lower:
            print(f"DEBUG: Detected keyword '{keyword}' in '{text_lower}'")
            return True
    print(f"DEBUG: No command keyword found in '{text_lower}'")
    return False

def get_command_from_llm(command_prompt):
    """
    Query the specialized command model to translate natural language
    into a structured system command. Expects a JSON-formatted response.
    """
    result = query_ollama(COMMAND_MODEL, command_prompt, stream=False)
    print("DEBUG: Command model raw result:", result)
    if result:
        response_text = result.get("response", "{}")
        print("DEBUG: Command model response text:", response_text)
        cleaned_text = clean_json_response(response_text)
        print("DEBUG: Cleaned command model response text:", cleaned_text)
        try:
            command_data = json.loads(cleaned_text)
            return command_data
        except json.JSONDecodeError:
            print("Failed to parse command response as JSON.")
            return None
    return None

def get_command_summary(command_str):
    """
    Use the conversational model to generate a one-sentence summary
    of what the proposed command does.
    """
    summary_prompt = f"Summarize in one sentence what the following command does: {command_str}"
    result = query_ollama(CONVERSATIONAL_MODEL, summary_prompt, stream=False)
    if result:
        summary = result.get("response", "")
        return summary
    return ""

# --- KDE Connect and Command Execution ---
def push_command_via_kde(command_text):
    """Push the command text to your phone using KDE Connect."""
    try:
        subprocess.run(["kdeconnect-cli", "--send-notification", f"Command: {command_text}"], check=True)
        print("Command pushed to phone for inspection.")
    except Exception as e:
        print("Error sending KDE Connect notification:", e)

def execute_command(command_str):
    """Execute the given command string on the local system."""
    try:
        subprocess.run(command_str.split(), check=True)
        print("Command executed successfully.")
    except Exception as e:
        print("Command execution failed:", e)

# --- Main Loop ---
def main_loop():
    print("Starting voice command interface...")
    while True:
        # Step 1: Capture audio and transcribe
        user_input = listen_audio()
        if not user_input:
            continue

        # Check if the input is a command before processing as conversation
        if detect_command(user_input):
            print("DEBUG: Command detected in input.")
            if voice_confirmation("Command detected. Do you want to issue this command? Please say yes or no.", timeout=15):
                # Here we include explicit instructions for KDE/Manjaro KDE:
                command_prompt = (
                    f"This system is running on {SYSTEM_INFO}. "
                    "Please convert the following instruction into a JSON object with a single key 'command' that is appropriate for a Manjaro KDE environment. "
                    "Do not output generic commands such as 'gnome-terminal'; instead, use 'konsole' or another KDE-compatible terminal. "
                    f"Instruction: {user_input}"
                )
                print(f"DEBUG: command prompt: {command_prompt}")
                command_details = get_command_from_llm(command_prompt)
                if command_details and "command" in command_details:
                    command_str = command_details["command"]
                    print("Command generated:", command_str)
                    text_to_speech(f"Proposed command: {command_str}")
                    summary = get_command_summary(command_str)
                    if summary:
                        print("Command summary:", summary)
                        text_to_speech(f"Summary: {summary}")
                    else:
                        print("No summary generated.")
                    if voice_confirmation("Do you want to execute the above command? Please say yes or no.", timeout=5):
                        execute_command(command_str)
                    else:
                        print("Command execution canceled.")
                else:
                    print("Could not generate a valid command.")
            else:
                print("Command not confirmed.")
            # Pause before listening again to allow interaction to finish.
            time.sleep(3)
            print("\n--- Waiting for next input ---\n")
            continue

        # Otherwise, process as normal conversation
        conv_response_data = query_ollama(CONVERSATIONAL_MODEL, user_input, stream=False)
        if conv_response_data:
            conv_response = conv_response_data.get("response", "")
            print("Conversational LLM response:", conv_response)
            text_to_speech(conv_response)
        else:
            print("No valid response from the conversational model.")

        print("\n--- Waiting for next input ---\n")

if __name__ == "__main__":
    main_loop()
