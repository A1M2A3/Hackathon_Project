
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


def load_player_count(default=4):
    """Number of players = max_people from config.yaml (falls back to 4)."""
    try:
        import yaml
        with open(os.path.join(_BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return int(cfg["camera"]["max_people"])
    except Exception:
        return default


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

        # Players
        self.num_players = load_player_count()
        self.name_vars = [
            tk.StringVar(value=f"Player {i}")
            for i in range(1, self.num_players + 1)
        ]
        self.survived = [False] * self.num_players
        self.eliminated = [False] * self.num_players

        # True while a snapshot is on screen (game paused on a detection)
        self.snapshot_showing = False

        self.build_main_menu()
        self.build_game_screen()
        self.build_results_screen()

        # Number keys mark players as survived during a game
        self.root.bind("<Key>", self.on_key_press)

        self.show_main_menu()

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
    # MAIN MENU
    # =========================================

    def build_main_menu(self):

        self.menu_frame = ttk.Frame(self.root, padding=30)

        ttk.Label(
            self.menu_frame,
            text="Cyclops",
            font=("Segoe UI", 32, "bold")
        ).pack(pady=(40, 20))

        # Player name entries
        players_box = ttk.LabelFrame(
            self.menu_frame,
            text="Players",
            padding=10
        )
        players_box.pack(pady=10)

        for i, var in enumerate(self.name_vars, start=1):
            ttk.Label(
                players_box,
                text=f"{i}.",
                font=("Segoe UI", 12)
            ).grid(row=i, column=0, padx=(0, 8), pady=3, sticky="e")

            ttk.Entry(
                players_box,
                textvariable=var,
                width=25
            ).grid(row=i, column=1, pady=3)

        # Menu buttons
        buttons = ttk.Frame(self.menu_frame)
        buttons.pack(pady=20)

        self.home_button = ttk.Button(
            buttons,
            text="Home Motor",
            command=self.home_motor
        )
        self.home_button.pack(fill="x", pady=5, ipadx=20, ipady=5)

        self.start_button = ttk.Button(
            buttons,
            text="Start Game",
            command=self.start_game,
            state="disabled"
        )
        self.start_button.pack(fill="x", pady=5, ipadx=20, ipady=5)

        self.quit_button = ttk.Button(
            buttons,
            text="Quit Game",
            command=self.close_window
        )
        self.quit_button.pack(fill="x", pady=5, ipadx=20, ipady=5)

        self.status_var = tk.StringVar(value="Home the motor to begin.")

        ttk.Label(
            self.menu_frame,
            textvariable=self.status_var,
            foreground="gray"
        ).pack(pady=10)

    # =========================================
    # GAME SCREEN
    # =========================================

    def build_game_screen(self):

        self.game_frame = ttk.Frame(self.root)

        # -------------------------------------
        # Player list (right side)
        # -------------------------------------

        panel = ttk.Frame(self.game_frame, padding=15)
        panel.pack(side="right", fill="y")

        ttk.Label(
            panel,
            text="Players",
            font=("Segoe UI", 16, "bold")
        ).pack(anchor="w", pady=(0, 10))

        self.player_rows = [
            self.make_player_row(panel, font_size=14)
            for _ in range(self.num_players)
        ]

        ttk.Label(
            panel,
            text=(
                f"Press 1-{self.num_players} to mark a player as survived.\n\n"
                f"While a snapshot is shown, press 1-{self.num_players} "
                "to eliminate that player."
            ),
            wraplength=180,
            foreground="gray"
        ).pack(side="bottom", anchor="w")

        # -------------------------------------
        # Camera display
        # -------------------------------------

        self.camera_label = ttk.Label(self.game_frame, anchor="center")

        self.camera_label.pack(
            side="left",
            fill="both",
            expand=True
        )

        # Resize image whenever window changes size
        self.camera_label.bind(
            "<Configure>",
            self.resize_camera
        )

        # -------------------------------------
        # Overlay controls
        # -------------------------------------

        self.controls = ttk.Frame(
            self.camera_label,
            padding=10
        )

        self.controls.place(
            relx=0.5,
            y=15,
            anchor="n"
        )

        self.continue_button = ttk.Button(
            self.controls,
            text="Continue Game",
            command=self.continue_game,
            state="disabled"
        )
        self.continue_button.pack(side="left", padx=5)

        self.end_button = ttk.Button(
            self.controls,
            text="End Game",
            command=self.end_game
        )
        self.end_button.pack(side="left", padx=5)

    # =========================================
    # RESULTS SCREEN
    # =========================================

    def build_results_screen(self):

        self.results_frame = ttk.Frame(self.root, padding=30)

        ttk.Label(
            self.results_frame,
            text="Game Over",
            font=("Segoe UI", 32, "bold")
        ).pack(pady=(40, 20))

        box = ttk.LabelFrame(
            self.results_frame,
            text="Results",
            padding=15
        )
        box.pack(pady=10)

        self.result_rows = [
            self.make_player_row(box, font_size=18)
            for _ in range(self.num_players)
        ]

        ttk.Button(
            self.results_frame,
            text="Return to Main Menu",
            command=self.show_main_menu
        ).pack(pady=30, ipadx=20, ipady=5)

    # =========================================
    # SCREEN SWITCHING
    # =========================================

    def show_main_menu(self):

        self.game_frame.pack_forget()
        self.results_frame.pack_forget()
        self.menu_frame.pack(fill="both", expand=True)

    def show_game_screen(self):

        self.menu_frame.pack_forget()
        self.results_frame.pack_forget()
        self.refresh_player_list()
        self.game_frame.pack(fill="both", expand=True)

    def show_results_screen(self):

        self.game_frame.pack_forget()
        self.menu_frame.pack_forget()
        self.refresh_rows(self.result_rows)
        self.results_frame.pack(fill="both", expand=True)

    # =========================================
    # PLAYER LIST
    # =========================================

    def make_player_row(self, parent, font_size):
        """A row with a player name, a status label and a red strike line."""

        row = ttk.Frame(parent)
        row.pack(anchor="w", fill="x", pady=4)

        name_label = tk.Label(row, font=("Segoe UI", font_size))
        name_label.pack(side="left")

        status_label = ttk.Label(
            row,
            font=("Segoe UI", font_size, "bold")
        )
        status_label.pack(side="left", padx=(10, 0))

        # Single red line drawn through the name when eliminated
        strike = tk.Frame(row, bg="red", height=2)

        return name_label, status_label, strike

    def player_name(self, index):

        name = self.name_vars[index].get().strip()
        return name or f"Player {index + 1}"

    def refresh_rows(self, rows):

        for i, (name_label, status_label, strike) in enumerate(rows):
            name_label.config(text=f"{i + 1}. {self.player_name(i)}")

            if self.eliminated[i]:
                status_label.config(text="Eliminated", foreground="red")
                strike.place(
                    in_=name_label,
                    relx=0, rely=0.5, relwidth=1,
                    height=2, anchor="w"
                )
                strike.lift()
            elif self.survived[i]:
                status_label.config(text="Survived", foreground="green")
                strike.place_forget()
            else:
                status_label.config(text="")
                strike.place_forget()

    def refresh_player_list(self):

        self.refresh_rows(self.player_rows)

    def on_key_press(self, event):

        # Only react on the game screen (so typing names isn't affected)
        if not self.game_frame.winfo_ismapped():
            return

        if not event.char or not event.char.isdigit():
            return

        index = int(event.char) - 1

        if not 0 <= index < self.num_players:
            return

        # Pressing again un-marks, in case of a mistake
        if self.snapshot_showing:
            self.eliminated[index] = not self.eliminated[index]
            if self.eliminated[index]:
                self.survived[index] = False
        else:
            self.survived[index] = not self.survived[index]
            if self.survived[index]:
                self.eliminated[index] = False

        self.refresh_player_list()

        if all(s or e for s, e in zip(self.survived, self.eliminated)):
            self.finish_game()

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
        self.status_var.set("Homing motor...")

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

        if not self.motor_homed or self.game_running:
            return

        self.stop_event.clear()
        self.continue_event.set()

        # Fresh game: nobody has survived yet, no old frames
        self.survived = [False] * self.num_players
        self.eliminated = [False] * self.num_players
        self.snapshot_showing = False
        self.latest_frame = None
        self.camera_photo = None
        self.camera_label.configure(image="")

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

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

        self.continue_button.config(state="disabled")
        self.show_game_screen()

        self.messages.put("Starting game...")

    # =========================================
    # END GAME
    # =========================================

    def stop_game(self):

        self.stop_event.set()

        # Release game if it is paused
        self.continue_event.set()

        self.snapshot_showing = False
        self.continue_button.config(state="disabled")

        if self.game_running:
            # Wait for "Game stopped" before allowing a new game
            self.start_button.config(state="disabled")
            self.home_button.config(state="disabled")
            self.status_var.set("Stopping game...")
            self.messages.put("Stopping game...")

    def end_game(self):

        self.stop_game()
        self.show_main_menu()

    def finish_game(self):
        """All players are resolved: stop the game and show the results."""

        self.stop_game()
        self.show_results_screen()

    # =========================================
    # CONTINUE GAME
    # =========================================

    def continue_game(self):

        self.snapshot_showing = False
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
                    self.snapshot_showing = True

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

                self.status_var.set("Motor homed. Ready to start.")

            elif message.startswith("Motor homing failed"):

                self.home_button.config(
                    state="normal"
                )

                self.status_var.set(message)

            elif (
                message.startswith("Person ")
                and "Waiting for Continue" in message
            ):

                self.continue_button.config(
                    state="normal"
                )

            elif message.startswith("Game resumed"):

                self.snapshot_showing = False

                self.continue_button.config(
                    state="disabled"
                )

            elif message.startswith("Game stopped"):

                self.game_running = False

                self.continue_button.config(
                    state="disabled"
                )

                self.home_button.config(
                    state="normal"
                )

                self.start_button.config(
                    state="normal" if self.motor_homed else "disabled"
                )

                self.status_var.set(
                    "Ready to start." if self.motor_homed
                    else "Home the motor to begin."
                )

            elif message.startswith("ERROR"):

                self.status_var.set(message)

        # -------------------------------------
        # 4. SCHEDULE NEXT UPDATE
        # -------------------------------------

        self.root.after(
            30,
            self.update_interface
        )

    # =========================================
    # CLOSE WINDOW / QUIT
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
