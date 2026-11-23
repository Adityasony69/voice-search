"""Hold Left Ctrl, speak, let go -> your words go into Windows search.

Runs fully offline with faster-whisper. Press Esc twice quickly to quit.
"""
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

# ---- settings ----
HOTKEY = keyboard.Key.ctrl_l   # hold this key to talk
MODEL_SIZE = "base.en"         # tiny.en (fastest) / base.en / small.en (most accurate)
PRESS_ENTER = True             # press Enter after typing the text
SAMPLE_RATE = 16000            # whisper expects 16 kHz mono
MIN_SECONDS = 0.4              # ignore taps shorter than this

print(f"Loading Whisper model '{MODEL_SIZE}' (first run downloads it)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

typer = keyboard.Controller()
chunks = []
recording = False
stream = None
last_esc = 0.0


def on_audio(indata, frames, time_info, status):
    if recording:
        chunks.append(indata.copy())


def transcribe(audio):
    segments, _ = model.transcribe(audio, language="en", beam_size=5, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def search_windows(text):
    typer.tap(keyboard.Key.cmd)   # open Start / search
    time.sleep(0.6)               # give the search box time to appear
    typer.type(text)
    if PRESS_ENTER:
        time.sleep(0.4)
        typer.tap(keyboard.Key.enter)


def handle(audio):
    if len(audio) < SAMPLE_RATE * MIN_SECONDS:
        print("  (too short, ignored)")
        return
    start = time.time()
    text = transcribe(audio)
    if not text:
        print("  (heard nothing)")
        return
    print(f"  heard: {text!r}  [{time.time() - start:.1f}s]")
    search_windows(text.rstrip("."))


def on_press(key):
    global recording, last_esc
    if key == HOTKEY and not recording:
        chunks.clear()
        recording = True
        print("Listening...")
    elif key == keyboard.Key.esc:
        now = time.time()
        if now - last_esc < 0.5:
            print("Bye.")
            return False          # stops the listener
        last_esc = now


def on_release(key):
    global recording
    if key == HOTKEY and recording:
        recording = False
        audio = np.concatenate(chunks).flatten() if chunks else np.zeros(0, np.float32)
        # do the slow work off the keyboard thread so keys don't lag
        threading.Thread(target=handle, args=(audio,), daemon=True).start()


def main():
    global stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=on_audio)
    stream.start()
    print("Ready. Hold Left Ctrl and speak. Double-tap Esc to quit.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    stream.stop()


if __name__ == "__main__":
    main()
