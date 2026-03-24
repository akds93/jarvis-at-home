# Jarvis-at-Home

A local, privacy-first AI voice assistant inspired by Jarvis from Iron Man.
Runs entirely on your own hardware — no cloud required except Google STT.

## Features

- **Wake word** — say "Hey Jarvis" to activate (powered by openwakeword)
- **Natural voice** — Piper TTS for human-sounding speech, pyttsx3 as fallback
- **Streaming responses** — speaks sentence by sentence as the LLM generates
- **Conversation memory** — remembers context across sessions
- **Jarvis personality** — calm, precise, dry wit, addresses you by name
- **System commands** — voice-controlled shell commands with double confirmation
- **System stats** — CPU, RAM, disk, temperature, battery on demand
- **Weather** — current weather via wttr.in (no API key needed)
- **Web search** — DuckDuckGo instant answers (no API key needed)
- **Media control** — play, pause, skip, volume via playerctl
- **Reminders** — "remind me in 10 minutes to..."
- **File search** — find files in your home directory by name
- **System monitor** — background alerts when CPU or disk usage is high
- **KDE Connect** — optional phone notifications before running commands

---

## Requirements

- **OS:** Linux (tested on Manjaro KDE)
- **Ollama** — running locally or on a remote host on your network
- **Python 3.10+**
- **Microphone** and **speakers**
- **Internet** — only for Google STT transcription and weather/search

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/akds93/jarvis-at-home.git
cd jarvis-at-home
```

### 2. Install system packages

```bash
bash install_req.bash
```

This installs: `espeak-ng`, `portaudio`, `playerctl`, `pulseaudio`, `pulsemixer`

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download a Piper voice model

Jarvis uses [Piper TTS](https://github.com/rhasspy/piper) for a natural voice.
Download a voice model from [rhasspy/piper-voices on Hugging Face](https://huggingface.co/rhasspy/piper-voices).

Recommended voice: `en_US-lessac-medium`

```bash
mkdir -p ~/.local/share/piper

# Download the model and its config
wget -P ~/.local/share/piper \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"

wget -P ~/.local/share/piper \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
```

If you skip this step, Jarvis will fall back to pyttsx3 (robotic voice).

### 5. Set up Ollama

Ensure [Ollama](https://ollama.com) is running and you have a model pulled:

```bash
ollama pull glm4-flash   # or any model you prefer
ollama list              # confirm the exact model name
```

### 6. Configure

Edit `config.yaml` in the project root:

```yaml
ollama:
  host: "http://YOUR_OLLAMA_IP:11434"   # change this
  conversational_model: "glm4-flash:latest"
  command_model: "glm4-flash:latest"
  classifier_model: "glm4-flash:latest"

tts:
  piper:
    model_path: "/home/YOUR_USERNAME/.local/share/piper/en_US-lessac-medium.onnx"

behavior:
  user_name: "Sir"   # how Jarvis addresses you
```

All other settings have sensible defaults — see `config.yaml` for the full reference.

### 7. Run

```bash
python3 code/jarvis-at-homeV5.py
```

Jarvis will greet you and wait for the wake word **"Hey Jarvis"**.

---

## Usage

| Say | What happens |
|---|---|
| "Hey Jarvis" | Activates listening |
| "Open the terminal" | Generates + confirms + runs the command |
| "How's the system?" | Reports CPU, RAM, disk, temp, battery |
| "What's the weather?" | Fetches weather for your location |
| "Search for the latest AI news" | DuckDuckGo instant answer |
| "Pause the music" | Runs `playerctl pause` |
| "Remind me in 5 minutes to drink water" | Sets a timed reminder |
| "Find my resume" | Searches home directory |
| Anything else | Conversation with memory |

---

## Configuration Reference

All settings live in `config.yaml`. Key options:

| Setting | Description |
|---|---|
| `ollama.host` | Ollama server URL |
| `ollama.conversational_model` | Model for chat |
| `ollama.command_model` | Model for shell command generation |
| `tts.engine` | `piper` or `pyttsx3` |
| `tts.piper.model_path` | Path to `.onnx` voice file |
| `wake_word.enabled` | Enable/disable wake word |
| `wake_word.model` | openwakeword model name |
| `wake_word.threshold` | Detection sensitivity (0.0–1.0) |
| `behavior.user_name` | Your name |
| `behavior.require_confirmation` | Voice confirm before running commands |
| `system_monitor.cpu_alert_percent` | Alert threshold for CPU |
| `kde_connect.enabled` | Push commands to phone |

---

## Troubleshooting

**No audio output** — Check that `espeak-ng` is installed and pyttsx3 works:
```bash
python3 -c "import pyttsx3; e=pyttsx3.init(); e.say('test'); e.runAndWait()"
```

**Piper not working** — Verify the model path in `config.yaml` and that both the `.onnx` and `.onnx.json` files were downloaded.

**Wake word not triggering** — Lower `wake_word.threshold` in config (try `0.3`). Or set `wake_word.enabled: false` for always-listen mode.

**Cannot reach Ollama** — Confirm the host and port in config, and that `ollama serve` is running.

**STT not working** — Google STT requires an internet connection. Check your network.
