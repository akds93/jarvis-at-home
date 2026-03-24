#!/usr/bin/env python3
"""
Jarvis-at-Home V5
Local AI voice assistant — wake word, natural TTS, streaming responses,
system stats, weather, web search, media control, reminders, file search.
"""
from __future__ import annotations

# =============================================================================
# Section 1 — Imports
# =============================================================================
import collections
import datetime
import json
import logging
import os
import pathlib
import platform
import queue
import re
import shlex
import signal
import subprocess
import threading
import time

import psutil
import requests
import speech_recognition as sr
import yaml

try:
    import distro
except ImportError:
    distro = None

# --- Optional: openwakeword ---
try:
    import numpy as np
    import pyaudio as _pyaudio
    from openwakeword.model import Model as _OWWModel
    _OWW_AVAILABLE = True
except ImportError:
    _OWW_AVAILABLE = False

# --- Optional: piper TTS ---
try:
    from piper import PiperVoice as _PiperVoice
    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False

# pyttsx3 is the unconditional fallback
import pyttsx3 as _pyttsx3

# =============================================================================
# Section 2 — Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("jarvis")

# =============================================================================
# Section 3 — Config
# =============================================================================
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_CONFIG_PATH = (_SCRIPT_DIR / ".." / "config.yaml").resolve()


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


cfg = _load_config()

# Ollama
OLLAMA_HOST      = cfg["ollama"]["host"].rstrip("/")
OLLAMA_CHAT_URL  = f"{OLLAMA_HOST}/api/chat"
OLLAMA_GEN_URL   = f"{OLLAMA_HOST}/api/generate"
CONV_MODEL       = cfg["ollama"]["conversational_model"]
CMD_MODEL        = cfg["ollama"]["command_model"]
CLASSIFIER_MODEL = cfg["ollama"].get("classifier_model", CMD_MODEL)
OLLAMA_TIMEOUT   = cfg["ollama"].get("request_timeout", 60)

# Behavior
USER_NAME            = cfg["behavior"].get("user_name", "Sir")
MAX_HISTORY          = cfg["behavior"].get("max_history", 20)
HISTORY_PATH         = pathlib.Path(cfg["behavior"]["history_path"]).expanduser()
REQUIRE_CONFIRMATION = cfg["behavior"].get("require_confirmation", True)
CONFIRMATION_PAUSE   = cfg["behavior"].get("confirmation_pause", 2)

# =============================================================================
# Section 4 — System Info
# =============================================================================
def _build_system_info() -> str:
    name = platform.system()
    if name == "Linux":
        dist = distro.name()    if distro else "Linux"
        ver  = distro.version() if distro else ""
        de   = (os.environ.get("XDG_CURRENT_DESKTOP")
                or os.environ.get("DESKTOP_SESSION")
                or "Unknown DE")
        return f"{name} ({dist} {ver}, {de})"
    return name


SYSTEM_INFO = _build_system_info()
log.info(f"System: {SYSTEM_INFO}")

# =============================================================================
# Section 5 — TTS Subsystem
# =============================================================================
_tts_lock      = threading.Lock()
_piper_voice   = None   # set if piper loads successfully
_pyttsx3_engine = None  # set as fallback


def _init_tts():
    global _piper_voice, _pyttsx3_engine
    engine = cfg["tts"].get("engine", "piper")

    if _PIPER_AVAILABLE and engine == "piper":
        model_path = pathlib.Path(cfg["tts"]["piper"]["model_path"])
        if model_path.exists():
            try:
                _piper_voice = _PiperVoice.load(str(model_path))
                log.info(f"Piper TTS loaded: {model_path.name}")
                return
            except Exception as e:
                log.warning(f"Piper load failed ({e}), falling back to pyttsx3.")
        else:
            log.warning(
                f"Piper model not found at {model_path}. "
                "Download from https://huggingface.co/rhasspy/piper-voices — falling back to pyttsx3."
            )

    # pyttsx3 fallback
    _pyttsx3_engine = _pyttsx3.init()
    voices = _pyttsx3_engine.getProperty("voices")
    vi = cfg["tts"]["pyttsx3"].get("voice_index", 0)
    if voices:
        _pyttsx3_engine.setProperty("voice", voices[max(0, min(vi, len(voices) - 1))].id)
    _pyttsx3_engine.setProperty("rate",   cfg["tts"]["pyttsx3"].get("rate", 170))
    _pyttsx3_engine.setProperty("volume", cfg["tts"]["pyttsx3"].get("volume", 1.0))
    log.info("pyttsx3 TTS initialised.")


def speak(text: str):
    """Thread-safe, blocking TTS call."""
    text = text.strip()
    if not text:
        return
    log.info(f"Jarvis: {text}")
    with _tts_lock:
        if _piper_voice is not None:
            _speak_piper(text)
        else:
            _speak_pyttsx3(text)


def _speak_piper(text: str):
    sample_rate = cfg["tts"]["piper"].get("sample_rate", 22050)
    pa  = _pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=_pyaudio.paInt16, channels=1,
            rate=sample_rate, output=True,
        )
        for chunk in _piper_voice.synthesize_stream_raw(text):
            stream.write(chunk)
        stream.stop_stream()
        stream.close()
    finally:
        pa.terminate()


def _speak_pyttsx3(text: str):
    _pyttsx3_engine.say(text)
    _pyttsx3_engine.runAndWait()


def speak_async(text: str):
    """Non-blocking speak — used by background threads (reminders, monitor)."""
    threading.Thread(target=speak, args=(text,), daemon=True).start()

# =============================================================================
# Section 6 — Persistent Conversation History
# =============================================================================
_JARVIS_SYSTEM_PROMPT = (
    f"You are Jarvis, a sophisticated personal AI assistant running on {SYSTEM_INFO}. "
    f"You are calm, precise, and have a dry wit. "
    f"You address the user as '{USER_NAME}'. "
    "Give concise, direct answers. Never refuse reasonable requests. "
    "Do not break character."
)

_history: list[dict] = []


def load_history():
    global _history
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_PATH.exists():
        try:
            loaded = json.loads(HISTORY_PATH.read_text())
            # Ensure system prompt is always the first entry
            if loaded and loaded[0].get("role") == "system":
                loaded[0]["content"] = _JARVIS_SYSTEM_PROMPT
                _history = loaded
            else:
                _history = [{"role": "system", "content": _JARVIS_SYSTEM_PROMPT}] + loaded
            log.info(f"Loaded {len(_history) - 1} history turns from {HISTORY_PATH}")
            return
        except Exception as e:
            log.warning(f"Could not load history ({e}), starting fresh.")
    _history = [{"role": "system", "content": _JARVIS_SYSTEM_PROMPT}]


def save_history():
    try:
        HISTORY_PATH.write_text(json.dumps(_history, indent=2))
    except OSError as e:
        log.error(f"Could not save history: {e}")


def _trim_history():
    """Keep system prompt + last MAX_HISTORY turn pairs."""
    if MAX_HISTORY <= 0:
        return
    system = _history[0]
    rest   = _history[1:]
    max_entries = MAX_HISTORY * 2
    if len(rest) > max_entries:
        _history[:] = [system] + rest[-max_entries:]

# =============================================================================
# Section 7 — Ollama HTTP Helpers
# =============================================================================
def _post_generate(prompt: str, model: str) -> str | None:
    try:
        r = requests.post(
            OLLAMA_GEN_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        if r.ok:
            return r.json().get("response", "").strip()
        log.error(f"Ollama generate {r.status_code}: {r.text[:200]}")
    except requests.ConnectionError:
        log.error(f"Cannot reach Ollama at {OLLAMA_HOST}")
    except Exception as e:
        log.error(f"Ollama generate error: {e}")
    return None


def _post_chat_streaming(messages: list[dict]):
    """Yield text tokens from a streaming /api/chat call."""
    try:
        r = requests.post(
            OLLAMA_CHAT_URL,
            json={"model": CONV_MODEL, "messages": messages, "stream": True},
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        for line in r.iter_lines():
            if not line:
                continue
            data  = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                break
    except requests.ConnectionError:
        log.error(f"Cannot reach Ollama at {OLLAMA_HOST}")
    except Exception as e:
        log.error(f"Ollama chat stream error: {e}")


def _clean_json(text: str) -> str:
    text = text.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

# =============================================================================
# Section 8 — Unified Intent Classifier
# =============================================================================
_CLASSIFY_PROMPT = """\
You are an intent classifier for a voice assistant.
Given the user utterance, return ONLY a JSON object with keys "intent" and "data".

Valid intents:
  command        — launch/run/open/close/kill a program or shell command
  conversation   — general chat, questions, opinions, anything else
  system_stats   — asking about CPU, RAM, disk, temperature, battery
  weather        — current weather or forecast
  web_search     — search the internet for factual information
  media_control  — play/pause/skip/volume for music or video
  set_reminder   — remind the user of something after a delay
  file_search    — find a file on the filesystem

Data field per intent:
  command        → the original instruction verbatim
  conversation   → ""
  system_stats   → ""
  weather        → city/location if mentioned, else ""
  web_search     → the search query string
  media_control  → one of: play | pause | next | prev | volume_up | volume_down | mute
  set_reminder   → JSON string: {{"message": "...", "seconds": N}}
  file_search    → filename or pattern to find

Return ONLY the JSON object. No explanation. No markdown.

Utterance: {utterance}"""

_FALLBACK_COMMAND_KEYWORDS = [
    "open", "launch", "execute", "run", "shutdown",
    "close", "kill", "start", "reboot",
]


def classify(text: str) -> tuple[str, str]:
    raw = _post_generate(_CLASSIFY_PROMPT.format(utterance=text), CLASSIFIER_MODEL)
    if raw:
        try:
            parsed = json.loads(_clean_json(raw))
            intent = str(parsed.get("intent", "conversation"))
            data   = str(parsed.get("data",   ""))
            log.info(f"Intent: {intent!r}  Data: {data!r}")
            return intent, data
        except (json.JSONDecodeError, AttributeError) as e:
            log.warning(f"Intent parse failed ({e}), using fallback.")
    return _fallback_classify(text)


def _fallback_classify(text: str) -> tuple[str, str]:
    tl = text.lower()
    if any(k in tl for k in _FALLBACK_COMMAND_KEYWORDS):
        return "command", text
    return "conversation", ""

# =============================================================================
# Section 9 — Command Generation & Execution
# =============================================================================
_CMD_PROMPT = (
    "System: {system_info}.\n"
    "Convert the following natural language instruction into a JSON object with a "
    "single key 'command' containing the exact shell command for this system. "
    "Prefer KDE-compatible applications (konsole, dolphin, kate, etc.). "
    "Return only the JSON — no explanation, no markdown.\n"
    "Instruction: {instruction}"
)

_cmd_history: collections.deque = collections.deque(maxlen=10)


def get_command(instruction: str) -> str | None:
    raw = _post_generate(
        _CMD_PROMPT.format(system_info=SYSTEM_INFO, instruction=instruction),
        CMD_MODEL,
    )
    if raw:
        try:
            data = json.loads(_clean_json(raw))
            return data.get("command")
        except json.JSONDecodeError:
            log.error(f"Command JSON parse failed: {raw!r}")
    return None


def get_command_summary(cmd: str) -> str:
    result = _post_generate(
        f"In one short sentence, what does this shell command do: {cmd}",
        CONV_MODEL,
    )
    return result or ""


def execute_command(cmd_str: str) -> bool:
    try:
        proc = subprocess.Popen(shlex.split(cmd_str))
        _cmd_history.append({"cmd": cmd_str, "proc": proc})
        log.info(f"Launched: {cmd_str}")
        return True
    except Exception as e:
        log.error(f"Execution failed: {e}")
        speak("Execution failed.")
        return False

# =============================================================================
# Section 10 — System Stats
# =============================================================================
def get_system_stats() -> str:
    parts = []

    cpu = psutil.cpu_percent(interval=1)
    parts.append(f"CPU at {cpu:.0f} percent.")

    ram = psutil.virtual_memory()
    parts.append(
        f"RAM: {ram.used / 1024**3:.1f} of {ram.total / 1024**3:.1f} gigabytes used."
    )

    disk_path = cfg["system_monitor"].get("disk_path", "/")
    disk = psutil.disk_usage(disk_path)
    parts.append(
        f"Disk: {disk.used / 1024**3:.1f} of {disk.total / 1024**3:.1f} gigabytes used "
        f"({disk.percent:.0f} percent)."
    )

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    t = entries[0].current
                    parts.append(f"{name.replace('_', ' ')} temperature: {t:.0f} degrees Celsius.")
                    break
    except Exception:
        pass

    try:
        batt = psutil.sensors_battery()
        if batt:
            status = "charging" if batt.power_plugged else "on battery"
            parts.append(f"Battery at {batt.percent:.0f} percent, {status}.")
    except Exception:
        pass

    return " ".join(parts)

# =============================================================================
# Section 11 — Weather
# =============================================================================
def get_weather(location: str = "") -> str:
    loc = location.strip() or cfg["weather"].get("location", "")
    try:
        r = requests.get(
            f"https://wttr.in/{loc}",
            params={"format": "3"},
            headers={"User-Agent": "curl/7.0"},
            timeout=10,
        )
        if r.ok:
            return r.text.strip()
        return "I couldn't retrieve the weather right now."
    except Exception as e:
        log.error(f"Weather error: {e}")
        return "Weather service is unavailable."

# =============================================================================
# Section 12 — Web Search
# =============================================================================
def web_search(query: str) -> str:
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=10,
        )
        data   = r.json()
        answer = (data.get("AbstractText")
                  or data.get("Answer")
                  or data.get("Definition")
                  or "")
        if not answer:
            topics = data.get("RelatedTopics", [])
            if topics and isinstance(topics[0], dict):
                answer = topics[0].get("Text", "")
        if answer:
            # Trim overly long answers
            if len(answer) > 400:
                answer = answer[:400].rsplit(" ", 1)[0] + "..."
            return answer
        return f"I couldn't find a direct answer for '{query}'."
    except Exception as e:
        log.error(f"Web search error: {e}")
        return "Web search is unavailable right now."

# =============================================================================
# Section 13 — Media Control
# =============================================================================
_MEDIA_COMMANDS: dict[str, list[str]] = {
    "play":         ["playerctl", "play"],
    "pause":        ["playerctl", "pause"],
    "next":         ["playerctl", "next"],
    "prev":         ["playerctl", "previous"],
    "stop":         ["playerctl", "stop"],
    "volume_up":    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
    "volume_down":  ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"],
    "mute":         ["pactl", "set-sink-mute",   "@DEFAULT_SINK@", "toggle"],
}


def media_control(action: str) -> str:
    cmd = _MEDIA_COMMANDS.get(action)
    if not cmd:
        return f"Unknown media action: {action}"
    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            log.warning(f"Media command returned {result.returncode}: {result.stderr.decode()}")
        return action.replace("_", " ").capitalize() + "."
    except FileNotFoundError:
        return "Media control requires playerctl and pulseaudio utilities."
    except Exception as e:
        log.error(f"Media error: {e}")
        return "Media control failed."

# =============================================================================
# Section 14 — Reminders
# =============================================================================
_reminder_timers: list[threading.Timer] = []


def set_reminder(message: str, seconds: float) -> str:
    seconds = max(1.0, min(float(seconds), 86400.0))

    def _fire():
        speak_async(f"Reminder, {USER_NAME}: {message}")

    t = threading.Timer(seconds, _fire)
    t.daemon = True
    t.start()
    _reminder_timers.append(t)

    if seconds < 60:
        duration = f"{int(seconds)} second{'s' if seconds != 1 else ''}"
    elif seconds < 3600:
        mins = int(seconds / 60)
        duration = f"{mins} minute{'s' if mins != 1 else ''}"
    else:
        hrs = seconds / 3600
        duration = f"{hrs:.1f} hours"

    return f"Reminder set for {duration}, {USER_NAME}."


def _cancel_all_reminders():
    for t in _reminder_timers:
        t.cancel()

# =============================================================================
# Section 15 — File Search
# =============================================================================
def file_search(pattern: str) -> str:
    home = str(pathlib.Path.home())
    try:
        result = subprocess.run(
            ["find", home, "-iname", f"*{pattern}*", "-type", "f", "-maxdepth", "8"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l for l in result.stdout.strip().splitlines() if l]
        if not lines:
            return f"No files matching '{pattern}' found in your home directory."
        names = [pathlib.Path(p).name for p in lines[:5]]
        suffix = f" and {len(lines) - 5} more" if len(lines) > 5 else ""
        return f"Found {len(lines)} file{'s' if len(lines) != 1 else ''}: {', '.join(names)}{suffix}."
    except subprocess.TimeoutExpired:
        return "File search timed out."
    except Exception as e:
        log.error(f"File search error: {e}")
        return "File search failed."

# =============================================================================
# Section 16 — KDE Connect
# =============================================================================
def push_to_phone(text: str):
    kc = cfg.get("kde_connect", {})
    if not kc.get("enabled"):
        return
    cmd = ["kdeconnect-cli", "--send-notification", f"Jarvis: {text}"]
    device_id = kc.get("device_id", "")
    if device_id:
        cmd += ["--device", device_id]
    try:
        subprocess.run(cmd, capture_output=True)
    except Exception as e:
        log.error(f"KDE Connect error: {e}")

# =============================================================================
# Section 17 — Background System Monitor
# =============================================================================
_shutdown_event = threading.Event()


def _system_monitor_loop():
    mon          = cfg.get("system_monitor", {})
    interval     = mon.get("interval", 60)
    cpu_thresh   = mon.get("cpu_alert_percent", 90.0)
    disk_thresh  = mon.get("disk_alert_percent", 90.0)
    disk_path    = mon.get("disk_path", "/")

    while not _shutdown_event.wait(timeout=interval):
        try:
            cpu = psutil.cpu_percent(interval=2)
            if cpu >= cpu_thresh:
                speak_async(f"Warning, {USER_NAME}. CPU usage is at {cpu:.0f} percent.")

            disk = psutil.disk_usage(disk_path).percent
            if disk >= disk_thresh:
                speak_async(f"Warning, {USER_NAME}. Disk usage is at {disk:.0f} percent.")
        except Exception as e:
            log.error(f"System monitor error: {e}")

# =============================================================================
# Section 18 — Streaming TTS Pipeline
# =============================================================================
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _stream_and_speak(messages: list[dict]) -> str:
    """
    Stream tokens from Ollama, buffer into sentences, speak each sentence
    as it completes. Returns the full response text.
    """
    full_response = ""
    buf           = ""

    for token in _post_chat_streaming(messages):
        full_response += token
        buf           += token
        while True:
            m = _SENTENCE_END.search(buf)
            if not m:
                break
            sentence = buf[:m.end()].strip()
            buf      = buf[m.end():]
            if sentence:
                speak(sentence)

    # Speak any remaining text after the stream ends
    if buf.strip():
        speak(buf.strip())

    return full_response

# =============================================================================
# Section 19 — STT Subsystem
# =============================================================================
_mic_in_use = threading.Event()


def listen_audio(timeout: int = None) -> str | None:
    if timeout is None:
        timeout = cfg["stt"].get("listen_timeout", 10)

    # Wait until microphone is free (wake word thread may be using it)
    while _mic_in_use.is_set():
        time.sleep(0.05)

    _mic_in_use.set()
    rec = sr.Recognizer()
    device_index = cfg["stt"].get("device_index", -1)
    mic_kwargs   = {} if device_index == -1 else {"device_index": device_index}

    try:
        with sr.Microphone(**mic_kwargs) as source:
            rec.adjust_for_ambient_noise(source, duration=cfg["stt"].get("calibration_duration", 0.4))
            log.info(f"Listening ({timeout}s)...")
            audio = rec.listen(source, timeout=timeout)
            text  = rec.recognize_google(audio)
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
    except Exception as e:
        log.error(f"STT error: {e}")
        return None
    finally:
        _mic_in_use.clear()


def voice_confirmation(prompt: str, timeout: int = None) -> bool:
    if timeout is None:
        timeout = cfg["stt"].get("confirmation_timeout", 5)
    speak(prompt)
    resp = listen_audio(timeout=timeout)
    if resp:
        return any(w in resp.lower() for w in
                   ["yes", "yeah", "yep", "sure", "run it", "do it", "confirm", "go", "ok", "okay"])
    return False

# =============================================================================
# Section 20 — Wake Word
# =============================================================================
_wake_event = threading.Event()


def _wake_word_loop():
    ww_cfg      = cfg.get("wake_word", {})
    model_name  = ww_cfg.get("model", "hey_jarvis")
    threshold   = ww_cfg.get("threshold", 0.5)
    chunk_dur   = ww_cfg.get("chunk_duration", 0.08)
    sample_rate = 16000
    chunk_size  = int(sample_rate * chunk_dur)

    try:
        oww = _OWWModel(wakeword_models=[model_name], inference_framework="onnx")
    except Exception as e:
        log.error(f"Wake word model failed to load ({e}) — falling back to always-listen.")
        _wake_event.set()
        return

    pa     = _pyaudio.PyAudio()
    stream = pa.open(
        rate=sample_rate, channels=1,
        format=_pyaudio.paInt16,
        input=True, frames_per_buffer=chunk_size,
    )
    log.info(f"Wake word active: say '{model_name.replace('_', ' ')}'")

    try:
        while not _shutdown_event.is_set():
            if _mic_in_use.is_set():
                # STT is active — drain buffer without processing
                stream.read(chunk_size, exception_on_overflow=False)
                continue

            raw   = stream.read(chunk_size, exception_on_overflow=False)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            oww.predict(audio)

            scores = oww.prediction_buffer.get(model_name, [0])
            if scores and max(scores) >= threshold:
                log.info("Wake word detected.")
                _wake_event.set()
                time.sleep(1.5)  # debounce
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

# =============================================================================
# Section 21 — Time-of-Day Greeting
# =============================================================================
def _time_greeting() -> str:
    h = datetime.datetime.now().hour
    if h < 12:
        period = "morning"
    elif h < 18:
        period = "afternoon"
    else:
        period = "evening"
    return f"Good {period}, {USER_NAME}. Jarvis online."

# =============================================================================
# Section 22 — Dispatch (Intent Router)
# =============================================================================
def dispatch(intent: str, data: str, raw_text: str):
    _history.append({"role": "user", "content": raw_text})
    _trim_history()

    if intent == "system_stats":
        reply = get_system_stats()
        speak(reply)

    elif intent == "weather":
        reply = get_weather(data)
        speak(reply)

    elif intent == "web_search":
        reply = web_search(data or raw_text)
        speak(reply)

    elif intent == "media_control":
        reply = media_control(data)
        speak(reply)

    elif intent == "set_reminder":
        try:
            rd = json.loads(data)
            reply = set_reminder(rd["message"], float(rd["seconds"]))
        except Exception:
            reply = f"I couldn't parse that reminder request, {USER_NAME}."
        speak(reply)

    elif intent == "file_search":
        reply = file_search(data or raw_text)
        speak(reply)

    elif intent == "command":
        _handle_command_flow(data or raw_text)
        reply = None  # command flow speaks for itself

    else:
        # conversation — stream response
        reply = _stream_and_speak(_history)

    if reply:
        _history.append({"role": "assistant", "content": reply})
    save_history()


def _handle_command_flow(instruction: str):
    if REQUIRE_CONFIRMATION:
        if not voice_confirmation(
            f"I detected a command request, {USER_NAME}. Shall I proceed? Say yes or no."
        ):
            speak("Cancelled.")
            return

    cmd_str = get_command(instruction)
    if not cmd_str:
        speak("I couldn't generate a command for that.")
        return

    log.info(f"Generated command: {cmd_str}")
    speak(f"Proposed command: {cmd_str}")

    summary = get_command_summary(cmd_str)
    if summary:
        speak(f"This will: {summary}")

    push_to_phone(cmd_str)

    if REQUIRE_CONFIRMATION:
        if not voice_confirmation("Shall I run it? Say yes or no."):
            speak("Command cancelled.")
            return

    if execute_command(cmd_str):
        speak("Done.")

    time.sleep(CONFIRMATION_PAUSE)

# =============================================================================
# Section 23 — Graceful Shutdown
# =============================================================================
def _shutdown(sig=None, frame=None):
    log.info("Shutting down...")
    _shutdown_event.set()
    _cancel_all_reminders()
    save_history()
    speak(f"Saving state. Goodbye, {USER_NAME}.")

# =============================================================================
# Section 24 — Main
# =============================================================================
def main():
    log.info(f"Jarvis-at-Home V5 | {SYSTEM_INFO}")
    log.info(f"Ollama: {OLLAMA_HOST}")
    log.info(f"Models: conv={CONV_MODEL}  cmd={CMD_MODEL}  classifier={CLASSIFIER_MODEL}")

    _init_tts()
    load_history()

    # Start background threads
    if cfg["system_monitor"].get("enabled", True):
        threading.Thread(target=_system_monitor_loop, daemon=True, name="sysmon").start()

    ww_cfg        = cfg.get("wake_word", {})
    wake_enabled  = ww_cfg.get("enabled", True) and _OWW_AVAILABLE
    always_listen = not wake_enabled

    if ww_cfg.get("enabled", True) and not _OWW_AVAILABLE:
        log.warning("openwakeword not installed — running in always-listen mode.")

    if wake_enabled:
        threading.Thread(target=_wake_word_loop, daemon=True, name="wakeword").start()

    speak(_time_greeting())

    while not _shutdown_event.is_set():
        try:
            if not always_listen:
                # Wait for wake word (check every 100ms for clean shutdown)
                while not _wake_event.wait(timeout=0.1):
                    if _shutdown_event.is_set():
                        return
                _wake_event.clear()
                speak("Yes?")

            user_input = listen_audio()
            if not user_input:
                continue

            intent, data = classify(user_input)
            dispatch(intent, data, user_input)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    main()
