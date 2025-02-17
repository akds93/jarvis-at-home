# jarvis-at-home (WORK IN PROGRESS)
Poor mans Personal Assistant with system access

# Quick Overview: Jarvis-At-Home Voice Command Interface
    
- Install the dependencies using:

```bash
  pip install -r requirements.txt
```

- Ensure system-level packages are installed (e.g., espeak-ng for TTS and kdeconnect for notifications).

## 3. Configure Ollama Models
Start the Ollama server (using ollama serve).
Create the conversational model using your conversational modelfile.
Create the command interpreter model using your command modelfile.
Verify models with ollama list to ensure they are available.
## 4. Running the Script
Run your main Python script
```bash
python3 jarvis-at-home.py
```
- The script will:
  - Capture and transcribe voice input.
  - Determine if the input is a normal conversation or a command.
  - Use TTS to speak responses and confirmation prompts.
  - Include system information in the command prompt for generating OS-appropriate commands.
  - Provide voice-based confirmation before executing any command.
  - Pause briefly after command interactions to avoid re-listening prematurely.
## 5. Interaction Flow
- Normal Conversation:
The script queries the conversational model and speaks the response.

- Command Execution:
  - Detects a command (e.g., "open the terminal").
  - Asks for voice confirmation.
  - Uses system information to generate a tailored command (e.g., "konsole" for KDE).
  - Prints and speaks the proposed command along with an AI-generated summary.
  - Waits for final voice confirmation before executing the command.
