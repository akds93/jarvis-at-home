#!/usr/bin/env bash
# System-level dependencies for Jarvis-at-Home
# Run once before pip install -r requirements.txt

sudo pacman -Sy --needed \
    espeak-ng \
    python-pyaudio \
    portaudio \
    playerctl \
    pulseaudio \
    pulsemixer
