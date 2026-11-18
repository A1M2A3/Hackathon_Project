
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import cv2
import threading
import queue
import os
import wave
import winsound

import numpy as np

BOOM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boom.wav")


def ensure_boom_sound(path=BOOM_PATH):
    """Generate a synthetic 'boom' WAV file if one doesn't already exist."""
    if os.path.exists(path):
        return
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Low sine sweep (120Hz -> 35Hz) for the body of the boom
    freq = 35 + 85 * np.exp(-t * 6)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * np.exp(-t * 3.5)
    # Short noise burst for the initial impact
    rng = np.random.default_rng(0)
    crack = rng.uniform(-1, 1, t.size) * np.exp(-t * 25)
    signal = body + 0.5 * crack
    signal /= np.max(np.abs(signal))
    samples = (signal * 0.9 * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Custom snapshot clip. Drop "snapshot_sound.mp3" or "snapshot_sound.wav"
# next to this file; falls back to the synthetic boom if neither exists.
SNAPSHOT_SOUND_CANDIDATES = [
    os.path.join(_BASE_DIR, "Dramatic Vine_Instagram Boom - Sound Effect (HD).mp3"),
    os.path.join(_BASE_DIR, "snapshot_sound.mp3"),
    os.path.join(_BASE_DIR, "snapshot_sound.wav"),
]

_mci_opened = False


def _mci(command):
    """Send a command to the Windows MCI player (plays mp3 with no extra deps)."""
    import ctypes
    err = ctypes.windll.winmm.mciSendStringW(command, None, 0, None)
    if err:
        buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.winmm.mciGetErrorStringW(err, buf, 256)
        raise RuntimeError(buf.value)


def play_boom():
    """Play the snapshot sound (custom clip if present) without blocking the UI."""
    global _mci_opened
    sound_path = next(
        (p for p in SNAPSHOT_SOUND_CANDIDATES if os.path.exists(p)), None
    )
    try:
        if sound_path and sound_path.lower().endswith(".mp3"):
            if not _mci_opened:
                _mci(f'open "{sound_path}" type mpegvideo alias snapshot_snd')
                _mci_opened = True
            # Restart from the beginning on every snapshot
            _mci("play snapshot_snd from 0")
        else:
            winsound.PlaySound(
                sound_path or BOOM_PATH,
                winsound.SND_FILENAME | winsound.SND_ASYNC,
            )
    except RuntimeError as e:
        print(f"Could not play snapshot sound: {e}")

from main_snapshot import main
from motor_test import Motor, COM_PORT, microsteps_per_deg


class CyclopsInterface:

    def __init__(self, root):

        self.root = root
        ensure_boom_sound()
        self.root.title("Cyclops Control Panel")
        self.root.geometry("1050x800")
        self.root.minsize(600, 400)

        self.motor = None

        self.motor_homed = False
        self.game_running = False

        self.stop_event = threading.Event()
        self.continue_event = threading.Event()
        self.continue_event.set()

        self.frame_queue = queue.Queue(maxsize=1)
        self.messages = queue.Queue()

        self.game_thread = None
        self.motor_thread = None

        # Store most recent frame
        self.latest_frame = None
        self.camera_photo = None

        # =========================================
        # CAMERA DISPLAY
        # =========================================

        self.camera_label = ttk.Label(root)

        self.camera_label.pack(
            fill="both",
            expand=True
        )

        # Resize image whenever window changes size
        self.camera_label.bind(
            "<Configure>",
            self.resize_camera
        )

        # =========================================
        # OVERLAY CONTROL PANEL
        # =========================================

        self.controls = ttk.Frame(
            self.camera_label,
            padding=10
        )

        self.controls.place(
            relx=0.5,
            y=15,
            anchor="n"
        )

        self.home_button = ttk.Button(
            self.controls,
            text="Motor Home",
            command=self.home_motor
        )
        self.home_button.pack(side="left", padx=5)

        self.start_button = ttk.Button(
            self.controls,
            text="Start Game",
            command=self.start_game,
            state="disabled"
        )
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ttk.Button(
            self.controls,
            text="Stop Game",
            command=self.stop_game,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=5)

        self.continue_button = ttk.Button(
            self.controls,
            text="Continue",
            command=self.continue_game,
            state="disabled"
        )
        self.continue_button.pack(side="left", padx=5)

        # =========================================
        # START UI UPDATE LOOP
        # =========================================

        self.root.after(
            30,
            self.update_interface
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_window
        )

    # =========================================
    # CAMERA RESIZING
    # =========================================

    def resize_camera(self, event=None):

        if self.latest_frame is not None:
            self.display_frame(self.latest_frame)

    def display_frame(self, frame):

        width = self.camera_label.winfo_width()
        height = self.camera_label.winfo_height()

        if width <= 1 or height <= 1:
            return

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame_rgb)

        # Stretch image to fill entire camera widget
        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(image)

        self.camera_label.configure(
            image=photo,
            anchor="center"
        )

        # Keep reference to prevent garbage collection
        self.camera_photo = photo

    # =========================================
    # MOTOR HOMING
    # =========================================

    def home_motor(self):

        self.home_button.config(state="disabled")
        self.start_button.config(state="disabled")

        self.motor_thread = threading.Thread(
            target=self.home_motor_worker,
            daemon=True
        )

        self.motor_thread.start()

    def home_motor_worker(self):

        try:

            motor = Motor(
                COM_PORT,
                0,
                microsteps_per_deg
            )

            motor.home()

            self.motor = motor

            self.messages.put(
                "Motor homing successful"
            )

        except Exception as e:

            self.messages.put(
                f"Motor homing failed: {e}"
            )

    # =========================================
    # START GAME
    # =========================================

    def start_game(self):

        if not self.motor_homed:
            return

        self.stop_event.clear()
        self.continue_event.set()

        self.game_thread = threading.Thread(
            target=main,
            args=(
                self.motor,
                self.stop_event,
                self.continue_event,
                self.frame_queue,
                self.messages
            ),
            daemon=True
        )

        self.game_thread.start()

        self.game_running = True

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.home_button.config(state="disabled")

        self.messages.put("Starting game...")

    # =========================================
    # STOP GAME
    # =========================================

    def stop_game(self):

        self.stop_event.set()

        # Release game if it is paused
        self.continue_event.set()

        self.stop_button.config(state="disabled")

        self.messages.put("Stopping game...")

    # =========================================
    # CONTINUE GAME
    # =========================================

    def continue_game(self):

        self.continue_event.set()

        self.continue_button.config(state="disabled")

    # =========================================
    # UPDATE INTERFACE
    # =========================================

    def update_interface(self):

        # -------------------------------------
        # 1. DISPLAY LIVE CAMERA FRAME
        # -------------------------------------

        try:

            frame = self.frame_queue.get_nowait()

            self.latest_frame = frame

            self.display_frame(frame)

        except queue.Empty:
            pass

        # -------------------------------------
        # 2. PROCESS MESSAGES
        # -------------------------------------

        while not self.messages.empty():

            message = self.messages.get_nowait()

            # Handle snapshot messages
            if isinstance(message, tuple):

                if message[0] == "snapshot":

                    snapshot = message[1]

                    self.latest_frame = snapshot

                    # Display snapshot in centre
                    self.display_frame(snapshot)

                    # Boom!
                    play_boom()

                continue

            # Print messages in terminal
            print(message)

            # ---------------------------------
            # 3. UPDATE BUTTON STATES
            # ---------------------------------

            if message == "Motor homing successful":

                self.motor_homed = True

                self.start_button.config(
                    state="normal"
                )

                self.home_button.config(
                    state="disabled"
                )

            elif message.startswith("Motor homing failed"):

                self.home_button.config(
                    state="normal"
                )

            elif (
                message.startswith("Person ")
                and "Waiting for Continue" in message
            ):

                self.continue_button.config(
                    state="normal"
                )

            elif message == "Game resumed":

                self.continue_button.config(
                    state="disabled"
                )

            elif message == "Game stopped":

                self.game_running = False

                self.stop_button.config(
                    state="disabled"
                )

                self.continue_button.config(
                    state="disabled"
                )

                self.home_button.config(
                    state="normal"
                )

                self.start_button.config(
                    state="normal" if self.motor_homed else "disabled"
                )

        # -------------------------------------
        # 4. SCHEDULE NEXT UPDATE
        # -------------------------------------

        self.root.after(
            30,
            self.update_interface
        )

    # =========================================
    # CLOSE WINDOW
    # =========================================

    def close_window(self):

        self.stop_event.set()
        self.continue_event.set()

        self.root.destroy()


# =============================================
# RUN APPLICATION
# =============================================

if __name__ == "__main__":

    root = tk.Tk()

    app = CyclopsInterface(root)

    root.mainloop()


