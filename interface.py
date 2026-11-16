
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import cv2
import threading
import queue

from main_snapshot import main
from motor_test import Motor, COM_PORT, microsteps_per_deg


class CyclopsInterface:

    def __init__(self, root):

        self.root = root
        self.root.title("Cyclops Control Panel")
        self.root.geometry("1050x800")

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

        # -----------------------------
        # TITLE
        # -----------------------------

        ttk.Label(
            root,
            text="CYCLOPS CONTROL PANEL",
            font=("Arial", 20, "bold")
        ).pack(pady=15)

        # -----------------------------
        # CONTROL BUTTONS
        # -----------------------------

        controls = ttk.Frame(root)
        controls.pack(pady=10)

        self.home_button = ttk.Button(
            controls,
            text="Motor Home",
            command=self.home_motor
        )
        self.home_button.pack(side="left", padx=10)

        self.start_button = ttk.Button(
            controls,
            text="Start Game",
            command=self.start_game,
            state="disabled"
        )
        self.start_button.pack(side="left", padx=10)

        self.stop_button = ttk.Button(
            controls,
            text="Stop Game",
            command=self.stop_game,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=10)

        self.continue_button = ttk.Button(
            controls,
            text="Continue",
            command=self.continue_game,
            state="disabled"
        )
        self.continue_button.pack(side="left", padx=10)

        # -----------------------------
        # STATUS
        # -----------------------------

        self.status_label = ttk.Label(
            root,
            text="Motor not homed | Game stopped",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=10)

        # -----------------------------
        # CAMERA DISPLAY
        # -----------------------------

        ttk.Label(
            root,
            text="Live Camera Feed",
            font=("Arial", 13, "bold")
        ).pack(pady=5)

        self.camera_label = ttk.Label(root)
        self.camera_label.pack(padx=10, pady=10)

        # -----------------------------
        # OUTPUT
        # -----------------------------

        ttk.Label(
            root,
            text="System Messages"
        ).pack(anchor="w", padx=20)

        self.output = tk.Text(
            root,
            height=7,
            width=110,
            state="disabled"
        )
        self.output.pack(padx=20, pady=10)

        self.root.after(30, self.update_interface)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_window
        )

    # =============================
    # MOTOR HOMING
    # =============================

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
            self.messages.put("Motor homing successful")

        except Exception as e:

            self.messages.put(f"Motor homing failed: {e}")

    # =============================
    # START GAME
    # =============================

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

    # =============================
    # STOP GAME
    # =============================

    def stop_game(self):

        self.stop_event.set()

        # Also release a paused game
        self.continue_event.set()

        self.stop_button.config(state="disabled")

        self.messages.put("Stopping game...")

    # =============================
    # CONTINUE GAME
    # =============================

    def continue_game(self):

        self.continue_event.set()

        self.continue_button.config(state="disabled")

    # =============================
    # UPDATE INTERFACE
    # =============================

    
    def update_interface(self):

        # -----------------------------
        # 1. DISPLAY LIVE CAMERA FRAME
        # -----------------------------

        try:
            frame = self.frame_queue.get_nowait()

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame)
            image.thumbnail((950, 520))

            photo = ImageTk.PhotoImage(image)

            self.camera_label.configure(image=photo)
            self.camera_label.image = photo

        except queue.Empty:
            pass

        # -----------------------------
        # 2. PROCESS MESSAGES
        # -----------------------------

        while not self.messages.empty():

            message = self.messages.get_nowait()

            # Handle snapshot messages separately
            if isinstance(message, tuple):

                if message[0] == "snapshot":

                    snapshot = message[1]

                    snapshot_rgb = cv2.cvtColor(
                        snapshot,
                        cv2.COLOR_BGR2RGB
                    )

                    image = Image.fromarray(snapshot_rgb)
                    image.thumbnail((950, 520))

                    photo = ImageTk.PhotoImage(image)

                    # Display snapshot in camera display widget
                    self.camera_label.configure(image=photo)
                    self.camera_label.image = photo

                continue

            # Handle ordinary text messages
            self.output.config(state="normal")
            self.output.insert("end", str(message) + "\n")
            self.output.see("end")
            self.output.config(state="disabled")

            # -----------------------------
            # 3. UPDATE BUTTON STATES
            # -----------------------------

            if message == "Motor homing successful":

                self.motor_homed = True
                self.start_button.config(state="normal")
                self.home_button.config(state="disabled")

            elif message.startswith("Motor homing failed"):

                self.home_button.config(state="normal")

            elif message.startswith("Person ") and "Waiting for Continue" in message:

                self.continue_button.config(state="normal")

            elif message == "Game resumed":

                self.continue_button.config(state="disabled")

            elif message == "Game stopped":

                self.game_running = False

                self.stop_button.config(state="disabled")
                self.continue_button.config(state="disabled")
                self.home_button.config(state="normal")

                self.start_button.config(
                    state="normal" if self.motor_homed else "disabled"
                )

        # -----------------------------
        # 4. UPDATE STATUS
        # -----------------------------

        motor_status = "Homed" if self.motor_homed else "Not homed"
        game_status = "Running" if self.game_running else "Stopped"

        self.status_label.config(
            text=f"Motor: {motor_status} | Game: {game_status}"
        )

        # Schedule next UI update
        self.root.after(30, self.update_interface)

        # =============================
    # CLOSE WINDOW
    # =============================

    def close_window(self):

        self.stop_event.set()
        self.continue_event.set()

        self.root.destroy()

if __name__ == "__main__":

    root = tk.Tk()

    app = CyclopsInterface(root)

    root.mainloop()
