#!/usr/bin/env python3
import json
import logging
import os
import platform
import shlex
import subprocess
import time

import requests
import speech_recognition as sr
import pyttsx3
import yaml

try:
    import distro
except ImportError:
    distro = None

from inputimeout import inputimeout, TimeoutOccurred

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("jarvis")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_SCRIPT_DIR, "..", "config.yaml")

def load_config():
    path = os.path.normpath(CONFIG_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

OLLAMA_HOST         = cfg["ollama"]["host"].rstrip("/")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_CHAT_URL     = f"{OLLAMA_HOST}/api/chat"
CONV_MODEL          = cfg["ollama"]["conversational_model"]
CMD_MODEL           = cfg["ollama"]["command_model"]
CLASSIFIER_MODEL    = cfg["ollama"].get("classifier_model", CMD_MODEL)

# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def get_system_info():
    os_name = platform.system()
    if os_name == "Linux":
        dist_name    = distro.name()    if distro else "Linux"
        dist_version = distro.version() if distro else ""
        de = (os.environ.get("XDG_CURRENT_DESKTOP")
              or os.environ.get("DESKTOP_SESSION")
              or "Unknown DE")
        return f"{os_name} ({dist_name} {dist_version}, {de})"
    return os_name

SYSTEM_INFO = get_system_info()
log.info(f"System: {SYSTEM_INFO}")

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
_tts = pyttsx3.init()
_voices = _tts.getProperty("voices")
_voice_idx = cfg["tts"].get("voice_index", 0)
if _voices:
    _voice_idx = max(0, min(_voice_idx, len(_voices) - 1))
    _tts.setProperty("voice", _voices[_voice_idx].id)
_tts.setProperty("rate",   cfg["tts"].get("rate",   170))
_tts.setProperty("volume", cfg["tts"].get("volume", 1.0))

def speak(text: str):
    log.info(f"Jarvis: {text}")
    _tts.say(text)
    _tts.runAndWait()

# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------
def listen_audio(timeout: int = None) -> str | None:
    if timeout is None:
        timeout = cfg["stt"].get("timeout", 15)
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        log.info(f"Listening (up to {timeout}s)...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
            text  = recognizer.recognize_google(audio)
            log.info(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            log.info("No speech detected.")
            return None
        except sr.UnknownValueError:
            log.info("Could not understand audio.")
            return None
        except sr.RequestError as e:
            log.error(f"STT service error: {e}")
            return None

def voice_confirmation(prompt: str, timeout: int = None) -> bool:
    if timeout is None:
        timeout = cfg["stt"].get("confirmation_timeout", 5)
    speak(prompt)
    response = listen_audio(timeout=timeout)
    if response:
        return any(w in response.lower() for w in ["yes", "yeah", "yep", "run it", "do it", "confirm", "go"])
    # Typed fallback
    try:
        typed = inputimeout(prompt="No voice detected. Type yes/no: ", timeout=10).strip().lower()
        return typed in ("yes", "y")
    except TimeoutOccurred:
        return False

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
def _post(url: str, payload: dict) -> dict | None:
    try:
        r = requests.post(url, json=payload, timeout=60)
        if r.ok:
            return r.json()
        log.error(f"Ollama error {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        log.error(f"Cannot reach Ollama at {OLLAMA_HOST}. Is it running?")
    except Exception as e:
        log.error(f"Ollama request failed: {e}")
    return None

def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

# ---------------------------------------------------------------------------
# Conversation (with history)
# ---------------------------------------------------------------------------
_history: list[dict] = []
_MAX_HISTORY = cfg["behavior"].get("max_history", 20)

def chat(user_input: str) -> str:
    _history.append({"role": "user", "content": user_input})
    # Trim history if needed (keep pairs: user + assistant)
    if _MAX_HISTORY > 0 and len(_history) > _MAX_HISTORY * 2:
        del _history[: len(_history) - _MAX_HISTORY * 2]
    result = _post(OLLAMA_CHAT_URL, {"model": CONV_MODEL, "messages": _history, "stream": False})
    if result:
        reply = result.get("message", {}).get("content", "").strip()
        _history.append({"role": "assistant", "content": reply})
        return reply
    return "I'm sorry, I didn't get a response from the language model."

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------
_CLASSIFY_PROMPT = (
    "Classify the following user input as either 'command' (a request to perform a "
    "system action: open/launch/run/close/shutdown something) or 'conversation' "
    "(anything else: questions, chat, opinions, etc.).\n"
    "Reply with exactly one word: command OR conversation.\n"
    "Input: {input}"
)

_FALLBACK_KEYWORDS = ["open", "launch", "execute", "run", "shutdown", "close", "kill", "start"]

def is_command(text: str) -> bool:
    result = _post(OLLAMA_GENERATE_URL, {
        "model":  CLASSIFIER_MODEL,
        "prompt": _CLASSIFY_PROMPT.format(input=text),
        "stream": False,
    })
    if result:
        reply = result.get("response", "").strip().lower()
        log.info(f"Intent classified as: {reply!r}")
        return "command" in reply
    # Fallback: keyword scan
    log.warning("Classifier unavailable, falling back to keyword detection.")
    return any(k in text.lower() for k in _FALLBACK_KEYWORDS)

# ---------------------------------------------------------------------------
# Command generation & execution
# ---------------------------------------------------------------------------
_CMD_PROMPT = (
    "System: {system_info}.\n"
    "Convert the following natural language instruction into a JSON object with a "
    "single key 'command' containing the exact shell command appropriate for this "
    "system. Use KDE-compatible applications (e.g. 'konsole' for terminal). "
    "Return only the JSON object — no explanation, no markdown.\n"
    "Instruction: {instruction}"
)

def get_command(user_input: str) -> str | None:
    prompt = _CMD_PROMPT.format(system_info=SYSTEM_INFO, instruction=user_input)
    result = _post(OLLAMA_GENERATE_URL, {"model": CMD_MODEL, "prompt": prompt, "stream": False})
    if result:
        raw     = result.get("response", "{}")
        cleaned = _clean_json(raw)
        try:
            data = json.loads(cleaned)
            return data.get("command")
        except json.JSONDecodeError:
            log.error(f"Failed to parse command JSON: {cleaned!r}")
    return None

def get_command_summary(command_str: str) -> str:
    result = _post(OLLAMA_GENERATE_URL, {
        "model":  CONV_MODEL,
        "prompt": f"In one short sentence, what does this shell command do: {command_str}",
        "stream": False,
    })
    if result:
        return result.get("response", "").strip()
    return ""

def execute_command(command_str: str) -> bool:
    try:
        args = shlex.split(command_str)
        subprocess.Popen(args)
        log.info(f"Launched: {command_str}")
        return True
    except Exception as e:
        log.error(f"Execution failed: {e}")
        speak("Sorry, the command failed to execute.")
        return False

# ---------------------------------------------------------------------------
# KDE Connect
# ---------------------------------------------------------------------------
def push_to_phone(text: str):
    kc = cfg.get("kde_connect", {})
    if not kc.get("enabled"):
        return
    cmd = ["kdeconnect-cli", "--send-notification", f"Jarvis: {text}"]
    device_id = kc.get("device_id", "")
    if device_id:
        cmd += ["--device", device_id]
    try:
        subprocess.run(cmd, check=True)
        log.info("Notification sent via KDE Connect.")
    except Exception as e:
        log.error(f"KDE Connect error: {e}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main_loop():
    log.info(f"Jarvis-at-Home starting | {SYSTEM_INFO}")
    log.info(f"Ollama host : {OLLAMA_HOST}")
    log.info(f"Conv model  : {CONV_MODEL}")
    log.info(f"Cmd model   : {CMD_MODEL}")
    speak("Jarvis online. How can I help?")

    while True:
        try:
            user_input = listen_audio()
            if not user_input:
                continue

            if is_command(user_input):
                # --- Command flow ---
                if not voice_confirmation(
                    "I detected a command request. Do you want to proceed? Say yes or no."
                ):
                    speak("Okay, cancelled.")
                    continue

                command_str = get_command(user_input)
                if not command_str:
                    speak("Sorry, I couldn't generate a command for that.")
                    continue

                log.info(f"Generated command: {command_str}")
                speak(f"Proposed command: {command_str}")

                summary = get_command_summary(command_str)
                if summary:
                    speak(f"This will: {summary}")

                push_to_phone(command_str)

                if voice_confirmation("Shall I run it? Say yes or no."):
                    if execute_command(command_str):
                        speak("Done.")
                else:
                    speak("Command cancelled.")

                time.sleep(cfg["behavior"].get("confirmation_pause", 3))

            else:
                # --- Conversation flow ---
                reply = chat(user_input)
                speak(reply)

        except KeyboardInterrupt:
            speak("Shutting down. Goodbye.")
            log.info("Exiting.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main_loop()
