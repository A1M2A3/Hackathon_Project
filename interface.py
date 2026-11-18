
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from functools import lru_cache

import cv2
import time
import random
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


# =========================================
# THEME (from the Figma main screen)
# =========================================

BG = "#0E0906"            # near-black brown background
PANEL = "#170F0A"         # button / entry fill
PANEL_HOVER = "#2A1A0E"
ORANGE = "#E8892B"        # accent: borders, small caps text
CREAM = "#F3E6CC"         # title / body text, eye white
RED = "#D8322A"           # iris
RED_DARK = "#7A1A14"
DISABLED = "#5A4A3A"
SURVIVED = "#9BC46A"      # muted green that reads on the dark background
MUTED = "#8A7F72"         # small grey captions on the camera screen
INK = "#1A120C"           # dark text on filled pills

# Outline / pill colour per player slot (P1, P2, ...)
PLAYER_COLORS = [ORANGE, "#4FC4B8", "#C38BE0", "#E8C547"]

# Cinzel font family bundled with the project (Cinzel/static/*.ttf)
CINZEL_DIR = os.path.join(_BASE_DIR, "Cinzel", "static")


def register_bundled_fonts():
    """Make the bundled Cinzel fonts usable by Tkinter (this process only).

    Uses the Windows FR_PRIVATE flag, so nothing is installed system-wide.
    Must run before the Tk root window is created.
    """
    if not os.path.isdir(CINZEL_DIR):
        return
    try:
        import ctypes
        FR_PRIVATE = 0x10
        for name in os.listdir(CINZEL_DIR):
            if name.lower().endswith(".ttf"):
                ctypes.windll.gdi32.AddFontResourceExW(
                    os.path.join(CINZEL_DIR, name), FR_PRIVATE, 0
                )
    except (AttributeError, OSError) as e:
        print(f"Could not load Cinzel fonts: {e}")


register_bundled_fonts()

# TrueType files for drawing on the camera image (first one found wins)
TITLE_FONT_FILES = [os.path.join(CINZEL_DIR, "Cinzel-Bold.ttf"),
                    "Cinzel-Bold.ttf", "georgiab.ttf", "DejaVuSerif-Bold.ttf"]
BODY_FONT_FILES = ["Inter-Regular.ttf", "segoeui.ttf", "DejaVuSans.ttf"]
BOLD_FONT_FILES = ["Inter-Bold.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"]


# Show the camera (and snapshots) mirrored, like a selfie view, so a player
# moving to their left moves left on screen.
MIRROR_CAMERA = True


def mirror_overlay(overlay, frame_w):
    """Flip the x coordinates of the detection overlay to match a mirror."""
    if not overlay:
        return overlay

    def flip(box):
        x1, y1, x2, y2 = box
        return (frame_w - x2, y1, frame_w - x1, y2)

    mirrored = dict(overlay)
    mirrored["people"] = [
        (track_id, flip(box), state)
        for track_id, box, state in overlay.get("people", [])
    ]
    if overlay.get("roi"):
        mirrored["roi"] = flip(overlay["roi"])
    return mirrored


@lru_cache(maxsize=64)
def load_font(files, size):
    """PIL font from the first TrueType file that loads."""
    for name in files:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=8)
def top_shade_mask(width, height):
    """Alpha mask that darkens the top of the camera image (for the title)."""
    mask = Image.new("L", (1, height))
    for y in range(height):
        mask.putpixel((0, y), int(235 * (1 - y / height) ** 1.4))
    return mask.resize((width, height))


def draw_dashed_rounded_rect(draw, box, radius, color, width=4,
                             dash=18, gap=12):
    """Dashed rounded rectangle (PIL has no dashed lines)."""
    x1, y1, x2, y2 = box
    r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    def dashed(ax, ay, bx, by):
        length = max(abs(bx - ax), abs(by - ay))
        if length <= 0:
            return
        pos = 0
        while pos < length:
            end = min(pos + dash, length)
            t0, t1 = pos / length, end / length
            draw.line(
                (ax + (bx - ax) * t0, ay + (by - ay) * t0,
                 ax + (bx - ax) * t1, ay + (by - ay) * t1),
                fill=color, width=width
            )
            pos += dash + gap

    dashed(x1 + r, y1, x2 - r, y1)
    dashed(x1 + r, y2, x2 - r, y2)
    dashed(x1, y1 + r, x1, y2 - r)
    dashed(x2, y1 + r, x2, y2 - r)
    if r:
        d = 2 * r
        draw.arc((x1, y1, x1 + d, y1 + d), 180, 270, fill=color, width=width)
        draw.arc((x2 - d, y1, x2, y1 + d), 270, 360, fill=color, width=width)
        draw.arc((x2 - d, y2 - d, x2, y2), 0, 90, fill=color, width=width)
        draw.arc((x1, y2 - d, x1 + d, y2), 90, 180, fill=color, width=width)


def draw_centered_text(draw, cx, y, text, font, fill):
    """Text horizontally centred on cx with its top at y; returns height."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (right - left) / 2 - left, y - top), text,
              font=font, fill=fill)
    return bottom - top


def pick_font(root, candidates):
    """First installed font family from candidates (last one is the fallback)."""
    import tkinter.font as tkfont
    installed = set(tkfont.families(root))
    return next((f for f in candidates if f in installed), candidates[-1])


def draw_greek_key(canvas, event=None):
    """Draw a repeating Greek key (meander) border across a canvas."""
    canvas.delete("all")
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    s = 4                      # grid size of one meander step
    top, bottom = 4, h - 4
    lw = 2
    canvas.create_line(0, top, w, top, fill=ORANGE, width=lw)
    canvas.create_line(0, bottom, w, bottom, fill=ORANGE, width=lw)
    y0 = top + s
    for x in range(0, w + 6 * s, 6 * s):
        pts = [
            (0, 5), (0, 0), (4, 0), (4, 3), (2, 3), (2, 2),
        ]
        coords = []
        for px, py in pts:
            coords += [x + px * s, y0 + py * s]
        canvas.create_line(*coords, fill=ORANGE, width=lw)
        canvas.create_line(x, y0 + 5 * s, x + 6 * s, y0 + 5 * s,
                           fill=ORANGE, width=lw)


# How far the iris can slide sideways, as a fraction of the eye's half-width
EYE_LOOK_RANGE = 0.42

# Motor travel that maps to the eye looking fully one way or the other
# (matches ANGLE_MIN / ANGLE_MAX in main_snapshot.get_detection_roi)
EYE_ANGLE_MIN = 0
EYE_ANGLE_MAX = 90


def eye_lid_curve(w, h, openness, sign, steps=32):
    """Points along an eyelid edge. sign -1 = upper lid, +1 = lower lid.

    At openness 1 the two lids trace the almond outline; at 0 they meet in
    a flat line through the middle.
    """
    cx, cy = w / 2, h / 2
    m = h * 0.055
    half = cx - m
    reach = (cy - m) * openness
    pts = []
    for i in range(steps + 1):
        x = m + 2 * half * i / steps
        t = (x - cx) / half
        pts.append((x, cy + sign * reach * (1 - t * t)))
    return pts


def eye_iris_x(w, h, look):
    """Iris centre x for look in [-1 (left), 1 (right)]."""
    half = w / 2 - h * 0.055
    return w / 2 + max(-1.0, min(1.0, look)) * half * EYE_LOOK_RANGE


def draw_eye(canvas, w, h, look=0.0, openness=1.0):
    """The big almond eye with a red iris, on a Tk canvas."""
    canvas.delete("all")
    cy = h / 2
    upper = eye_lid_curve(w, h, 1.0, -1)
    lower = eye_lid_curve(w, h, 1.0, 1)
    canvas.create_polygon(*[c for p in upper + lower[::-1] for c in p],
                          fill=CREAM, outline="")

    ix = eye_iris_x(w, h, look)
    r = h * 0.30
    canvas.create_oval(ix - r, cy - r, ix + r, cy + r,
                       fill=RED, outline=RED_DARK, width=3)
    p = r * 0.45
    canvas.create_oval(ix - p, cy - p, ix + p, cy + p,
                       fill="#0A0A0A", outline="")
    g = r * 0.14
    gx, gy = ix - p * 0.6, cy - p * 0.6
    canvas.create_oval(gx - g, gy - g, gx + g, gy + g,
                       fill=CREAM, outline="")

    # The lids also mask off any iris that pokes outside the almond
    for sign in (-1, 1):
        edge = 0 if sign < 0 else h
        curve = [c for p in eye_lid_curve(w, h, openness, sign) for c in p]
        canvas.create_polygon(0, edge, 0, cy, *curve, w, cy, w, edge,
                              fill=BG, outline="")
        canvas.create_line(*curve, fill=ORANGE, width=max(2, h / 30),
                           capstyle="round")


class EyeAnimator:
    """Main-menu eye: glances side to side and blinks every few seconds."""

    TICK_MS = 30
    BLINK_SECONDS = 0.24

    def __init__(self, canvas, w, h):
        self.canvas, self.w, self.h = canvas, w, h
        now = time.monotonic()
        self.look = 0.0
        self.target = 0.0
        self.openness = 1.0
        self.next_glance = now + random.uniform(0.8, 2.0)
        self.next_blink = now + random.uniform(1.2, 2.5)
        self.blink_start = None
        self.drawn = None
        self.tick()

    def tick(self):
        if not self.canvas.winfo_exists():
            return
        now = time.monotonic()

        # Pick somewhere new to look every so often
        if now >= self.next_glance:
            choices = [c for c in (-1.0, -0.6, 0.0, 0.6, 1.0)
                       if c != self.target]
            self.target = random.choice(choices)
            self.next_glance = now + random.uniform(1.2, 3.5)

        # Ease towards the target
        self.look += (self.target - self.look) * 0.18
        if abs(self.target - self.look) < 0.005:
            self.look = self.target

        # Blink: open -> shut (held briefly) -> open
        if self.blink_start is None and now >= self.next_blink:
            self.blink_start = now
        if self.blink_start is not None:
            t = (now - self.blink_start) / self.BLINK_SECONDS
            if t >= 1:
                self.openness = 1.0
                self.blink_start = None
                self.next_blink = now + random.uniform(2.5, 5.5)
            else:
                self.openness = max(0.0, min(1.0, abs(t - 0.5) * 2.4 - 0.2))

        state = (round(self.look, 3), round(self.openness, 3))
        if state != self.drawn and self.canvas.winfo_ismapped():
            draw_eye(self.canvas, self.w, self.h, self.look, self.openness)
            self.drawn = state

        self.canvas.after(self.TICK_MS, self.tick)


@lru_cache(maxsize=256)
def render_eye_image(w, h, look, closed=False):
    """The same eye as an RGBA PIL image (for drawing onto the camera feed).

    Rendered at 3x and scaled down for smooth edges. `look` should be
    rounded by the caller so the cache stays useful. `closed` draws the
    shut eye: a sagging lid line with lashes.
    """
    s = 3
    W, H = w * s, h * s
    cy = H / 2

    if closed:
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        lw = max(2, int(H / 30))
        lid = eye_lid_curve(W, H, 0.3, 1)
        d.line(lid, fill=ORANGE, width=lw, joint="curve")
        for x, y in (lid[0], lid[-1]):
            d.ellipse((x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2),
                      fill=ORANGE)
        # Lashes hanging off the lid, splayed outwards from the centre
        lash = H * 0.16
        for i in (6, 11, 16, 21, 26):
            x, y = lid[i]
            tilt = (x - W / 2) / (W / 2)
            d.line((x, y, x + tilt * lash * 0.6, y + lash),
                   fill=ORANGE, width=max(1, int(lw * 0.8)))
        return img.resize((w, h), Image.Resampling.LANCZOS)

    upper = eye_lid_curve(W, H, 1.0, -1)
    lower = eye_lid_curve(W, H, 1.0, 1)

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(upper + lower[::-1], fill=255)

    layer = Image.new("RGBA", (W, H), CREAM)
    d = ImageDraw.Draw(layer)
    ix = eye_iris_x(W, H, look)
    r = H * 0.30
    d.ellipse((ix - r, cy - r, ix + r, cy + r), fill=RED,
              outline=RED_DARK, width=max(1, int(H / 50)))
    p = r * 0.45
    d.ellipse((ix - p, cy - p, ix + p, cy + p), fill="#0A0A0A")
    g = r * 0.14
    gx, gy = ix - p * 0.6, cy - p * 0.6
    d.ellipse((gx - g, gy - g, gx + g, gy + g), fill=CREAM)

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    img.paste(layer, (0, 0), mask)

    d = ImageDraw.Draw(img)
    lw = max(2, int(H / 30))
    for curve in (upper, lower):
        d.line(curve, fill=ORANGE, width=lw, joint="curve")
    for x, y in (upper[0], upper[-1]):  # round the corners
        d.ellipse((x - lw / 2, y - lw / 2, x + lw / 2, y + lw / 2),
                  fill=ORANGE)

    return img.resize((w, h), Image.Resampling.LANCZOS)


def motor_angle_to_look(angle):
    """Map the motor angle to an on-screen eye direction in [-1, 1].

    The eye looks towards where the detection zone sits on screen, so the
    direction flips when the camera view is mirrored.
    """
    span = EYE_ANGLE_MAX - EYE_ANGLE_MIN
    fraction = max(0.0, min(1.0, (angle - EYE_ANGLE_MIN) / span))
    look = fraction * 2 - 1
    return -look if MIRROR_CAMERA else look


class HexButton(tk.Canvas):
    """Elongated-hexagon button in the Figma style.

    Supports .config(state="normal"/"disabled") like a regular button.
    """

    def __init__(self, parent, text, command, width=200, height=42,
                 font=("Georgia", 14), diamonds=False, state="normal"):
        self._pad = 34 if diamonds else 2
        super().__init__(
            parent,
            width=width + 2 * self._pad,
            height=height,
            bg=BG,
            highlightthickness=0,
            cursor="hand2"
        )
        self._text = text
        self._command = command
        self._btn_w = width
        self._btn_h = height
        self._font = font
        self._diamonds = diamonds
        self._state = state
        self._hover = False
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", self._on_click)
        self._draw()

    def configure(self, cnf=None, **kw):
        if "state" in kw:
            self._state = kw.pop("state")
            self._draw()
            if not kw and cnf is None:
                return None
        return super().configure(cnf, **kw)

    config = configure

    def _set_hover(self, value):
        self._hover = value
        self._draw()

    def _on_click(self, event):
        if self._state != "disabled" and self._command:
            self._command()

    def _draw(self):
        self.delete("all")
        enabled = self._state != "disabled"
        line = ORANGE if enabled else DISABLED
        text = CREAM if enabled else DISABLED
        fill = PANEL_HOVER if (self._hover and enabled) else PANEL
        self.configure(cursor="hand2" if enabled else "arrow")

        x0, x1 = self._pad, self._pad + self._btn_w
        h = self._btn_h
        mid = h / 2
        cut = 14
        self.create_polygon(
            x0, mid, x0 + cut, 2, x1 - cut, 2,
            x1, mid, x1 - cut, h - 2, x0 + cut, h - 2,
            fill=fill, outline=line, width=2
        )
        self.create_text((x0 + x1) / 2, mid, text=self._text,
                         fill=text, font=self._font)

        if self._diamonds:
            for dx in (self._pad / 2, x1 + self._pad / 2):
                d = 8
                self.create_polygon(dx, mid - d, dx + d, mid, dx, mid + d,
                                    dx - d, mid, fill=BG, outline=line,
                                    width=2)
                d = 3
                self.create_polygon(dx, mid - d, dx + d, mid, dx, mid + d,
                                    dx - d, mid, fill=line, outline="")


class CyclopsInterface:

    def __init__(self, root):

        self.root = root
        ensure_boom_sound()
        self.root.title("Nobody's Watching")
        self.root.geometry("1050x800")
        self.root.minsize(600, 400)

        self.motor = None

        self.motor_homed = False
        self.game_running = False

        self.stop_event = threading.Event()
        self.continue_event = threading.Event()
        self.continue_event.set()

        # Set by "All Players Ready": ends the lobby and starts the game
        self.ready_event = threading.Event()
        self.in_lobby = False

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

        # Snapshots taken during the current game (shown in the graveyard)
        self.game_snapshots = []
        # Player indices eliminated on each snapshot (parallel list)
        self.game_snapshot_players = []
        # Index into game_snapshots of the snapshot currently on screen
        self.current_snapshot = None
        self.graveyard_photos = []

        # Overlay data that came with the latest live frame
        self.latest_overlay = None
        # Track ID -> [player slot, last seen time] for the P1/P2 labels
        self.track_slots = {}

        # Fonts shared by the themed screens
        self._results_serif = pick_font(
            self.root, ["Cinzel", "Trajan Pro", "Cormorant Garamond", "Georgia"]
        )
        self._results_sans = pick_font(self.root, ["Inter", "Segoe UI"])

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

        serif = pick_font(
            self.root, ["Cinzel", "Trajan Pro", "Cormorant Garamond", "Georgia"]
        )
        sans = pick_font(self.root, ["Inter", "Segoe UI"])

        self.menu_frame = tk.Frame(self.root, bg=BG)

        # Greek key borders, top and bottom
        for side in ("top", "bottom"):
            border = tk.Canvas(self.menu_frame, height=32, bg=BG,
                               highlightthickness=0)
            border.pack(side=side, fill="x", padx=8, pady=8)
            border.bind("<Configure>",
                        lambda e, c=border: draw_greek_key(c, e))

        content = tk.Frame(self.menu_frame, bg=BG)
        content.pack(expand=True)

        tk.Label(
            content, text="NOBODY'S WATCHING", bg=BG, fg=ORANGE,
            font=(sans, 10, "bold")
        ).pack(pady=(0, 14))

        eye = tk.Canvas(content, width=380, height=150, bg=BG,
                        highlightthickness=0)
        eye.pack()
        self.menu_eye = EyeAnimator(eye, 380, 150)

        tk.Label(
            content, text="NOBODY'S WATCHING", bg=BG, fg=CREAM,
            font=(serif, 44, "bold")
        ).pack(pady=(26, 6))

        tk.Label(
            content,
            text="Move when the eye is closed to safely escape the cave.",
            bg=BG, fg=CREAM, font=(sans, 13)
        ).pack(pady=(0, 24))

        # Player name entries: "1. [entry]  2. [entry] ..."
        players_box = tk.Frame(content, bg=BG)
        players_box.pack(pady=(0, 24))

        for i, var in enumerate(self.name_vars, start=1):
            tk.Label(
                players_box, text=f"{i}.", bg=BG, fg=ORANGE,
                font=(sans, 12, "bold")
            ).grid(row=0, column=2 * (i - 1),
                   padx=(14 if i > 1 else 0, 4), pady=3, sticky="e")

            tk.Entry(
                players_box, textvariable=var, width=13,
                bg=PANEL, fg=CREAM, insertbackground=CREAM,
                relief="flat", font=(sans, 11),
                highlightthickness=1, highlightbackground=DISABLED,
                highlightcolor=ORANGE
            ).grid(row=0, column=2 * (i - 1) + 1, pady=3, ipady=4)

        # Main START button with diamonds either side
        self.start_button = HexButton(
            content, "START", self.start_game, width=200, height=44,
            font=(serif, 15), diamonds=True, state="disabled"
        )
        self.start_button.pack()

        # Secondary buttons
        secondary = tk.Frame(content, bg=BG)
        secondary.pack(pady=(12, 0))

        self.home_button = HexButton(
            secondary, "HOME MOTOR", self.home_motor, width=150, height=34,
            font=(serif, 11)
        )
        self.home_button.pack(side="left", padx=6)

        self.quit_button = HexButton(
            secondary, "QUIT", self.close_window, width=150, height=34,
            font=(serif, 11)
        )
        self.quit_button.pack(side="left", padx=6)

        tk.Label(
            content, text="STAND BACK SO THE CAMERA CAN SEE YOU",
            bg=BG, fg=ORANGE, font=(sans, 8)
        ).pack(pady=(18, 4))

        self.status_var = tk.StringVar(value="Home the motor to begin.")

        tk.Label(
            content, textvariable=self.status_var, bg=BG, fg=DISABLED,
            font=(sans, 9)
        ).pack()

    # =========================================
    # GAME SCREEN
    # =========================================

    def build_game_screen(self):

        serif = self._results_serif
        sans = self._results_sans

        self.game_frame = tk.Frame(self.root, bg=BG)

        # -------------------------------------
        # Bottom bar: player list, controls, key hints
        # -------------------------------------

        bar = tk.Frame(self.game_frame, bg=BG, padx=15, pady=10)
        bar.pack(side="bottom", fill="x")

        players_bar = tk.Frame(bar, bg=BG)
        players_bar.pack(side="top")

        self.player_rows = [
            self.make_player_row(players_bar, font_size=14, horizontal=True,
                                 themed=True)
            for _ in range(self.num_players)
        ]

        controls = tk.Frame(bar, bg=BG)
        controls.pack(side="top", pady=(8, 4))

        # Lobby only: starts the real game once everyone is lined up
        self.ready_button = HexButton(
            controls, "ALL PLAYERS READY", self.all_players_ready,
            width=230, height=38, font=(serif, 12), diamonds=True
        )

        self.continue_button = HexButton(
            controls, "CONTINUE", self.continue_game,
            width=170, height=38, font=(serif, 12), diamonds=True,
            state="disabled"
        )

        self.end_button = HexButton(
            controls, "END GAME", self.end_game,
            width=170, height=38, font=(serif, 12)
        )

        self.game_hint_var = tk.StringVar()

        tk.Label(
            bar,
            textvariable=self.game_hint_var,
            bg=BG, fg=MUTED, font=(sans, 8)
        ).pack(side="top")

        self.show_game_controls(lobby=False)

        # -------------------------------------
        # Camera display (title and player outlines are drawn onto it)
        # -------------------------------------

        self.camera_label = tk.Label(
            self.game_frame, bg=BG, bd=0, anchor="center"
        )

        self.camera_label.pack(
            side="top",
            fill="both",
            expand=True
        )

        # Resize image whenever window changes size
        self.camera_label.bind(
            "<Configure>",
            self.resize_camera
        )

    def show_game_controls(self, lobby):
        """Lobby: [All Players Ready][End Game]. Game: [Continue][End Game]."""

        for button in (self.ready_button, self.continue_button,
                       self.end_button):
            button.pack_forget()

        first = self.ready_button if lobby else self.continue_button
        first.pack(side="left", padx=6)
        self.end_button.pack(side="left", padx=6)

        if lobby:
            self.game_hint_var.set(
                "LINE UP SO EVERY PLAYER HAS AN OUTLINE, "
                "THEN PRESS ALL PLAYERS READY TO START"
            )
        else:
            self.game_hint_var.set(
                f"PRESS 1-{self.num_players} TO MARK A PLAYER AS SURVIVED"
                "   ·   "
                f"WHILE A SNAPSHOT IS SHOWN, PRESS 1-{self.num_players} "
                "TO ELIMINATE THAT PLAYER"
            )

    def all_players_ready(self):
        """Everyone is lined up: leave the lobby and start the game."""

        if not self.in_lobby or not self.game_running:
            return

        self.in_lobby = False
        self.ready_event.set()
        self.show_game_controls(lobby=False)

    # =========================================
    # RESULTS SCREEN
    # =========================================

    def build_results_screen(self):

        serif = self._results_serif
        sans = self._results_sans

        # Dark scrollbar borrowed from the "clam" theme so the rest of the
        # app keeps its native look.
        style = ttk.Style(self.root)
        for element in ("trough", "thumb"):
            try:
                style.element_create(
                    f"Dark.Horizontal.Scrollbar.{element}", "from", "clam"
                )
            except tk.TclError:
                pass  # already created
        style.layout("Dark.Horizontal.TScrollbar", [
            ("Dark.Horizontal.Scrollbar.trough", {
                "sticky": "we",
                "children": [("Dark.Horizontal.Scrollbar.thumb",
                              {"expand": "1", "sticky": "nswe"})]
            })
        ])
        style.configure(
            "Dark.Horizontal.TScrollbar",
            troughcolor=BG, background=ORANGE, bordercolor=BG,
            lightcolor=ORANGE, darkcolor=ORANGE, gripcount=0, arrowsize=6
        )

        self.results_frame = tk.Frame(self.root, bg=BG)

        # Greek key borders, top and bottom
        for side in ("top", "bottom"):
            border = tk.Canvas(self.results_frame, height=32, bg=BG,
                               highlightthickness=0)
            border.pack(side=side, fill="x", padx=8, pady=8)
            border.bind("<Configure>",
                        lambda e, c=border: draw_greek_key(c, e))

        content = tk.Frame(self.results_frame, bg=BG)
        content.pack(side="top", fill="both", expand=True, padx=30)

        tk.Label(
            content, text="NOBODY'S WATCHING", bg=BG, fg=ORANGE,
            font=(sans, 10, "bold")
        ).pack(pady=(4, 6))

        tk.Label(
            content, text="GAME OVER", bg=BG, fg=CREAM,
            font=(serif, 36, "bold")
        ).pack(pady=(0, 10))

        box = tk.Frame(content, bg=BG)
        box.pack(pady=(0, 6))

        self.result_rows = [
            self.make_player_row(box, font_size=12, horizontal=True,
                                 themed=True)
            for _ in range(self.num_players)
        ]

        # Button and graveyard are packed from the bottom up so the
        # snapshots sit at the bottom of the screen, above the button.
        HexButton(
            content, "RETURN TO MAIN MENU", self.show_main_menu,
            width=260, height=42, font=(serif, 13), diamonds=True
        ).pack(side="bottom", pady=(12, 4))

        # -------------------------------------
        # Graveyard: snapshots from this game
        # -------------------------------------

        graveyard = tk.Frame(
            content, bg=BG,
            highlightthickness=1, highlightbackground=ORANGE,
            padx=10, pady=8
        )
        # Expand into whatever vertical space is left so the thumbnails
        # can be sized to fit it.
        graveyard.pack(side="bottom", pady=(14, 4), fill="both", expand=True)

        tk.Label(
            graveyard, text="GRAVEYARD", bg=BG, fg=ORANGE,
            font=(serif, 14, "bold")
        ).pack(side="top", pady=(0, 6))

        # Scrollbar packed first so it is never squeezed out by the canvas
        graveyard_scroll = ttk.Scrollbar(
            graveyard, orient="horizontal", style="Dark.Horizontal.TScrollbar"
        )
        graveyard_scroll.pack(side="bottom", fill="x", pady=(6, 0))

        # Small requested height: the canvas takes the leftover space
        # instead of pushing other widgets (and itself) off screen.
        self.graveyard_canvas = tk.Canvas(
            graveyard,
            height=1,
            bg=BG,
            highlightthickness=0,
            xscrollcommand=graveyard_scroll.set
        )
        self.graveyard_canvas.pack(side="top", fill="both", expand=True)
        graveyard_scroll.configure(command=self.graveyard_canvas.xview)

        self.graveyard_inner = tk.Frame(self.graveyard_canvas, bg=BG)
        self.graveyard_window = self.graveyard_canvas.create_window(
            (0, 0), window=self.graveyard_inner, anchor="nw"
        )
        self.graveyard_inner.bind("<Configure>", self.center_graveyard)
        self.graveyard_canvas.bind("<Configure>", self.on_graveyard_resize)

        self._graveyard_thumb_h = None
        self._graveyard_resize_job = None

    # Upper limit for thumbnail height; actual height shrinks to fit.
    GRAVEYARD_THUMB_HEIGHT = 700
    # Room reserved under each thumbnail for its "#n" caption.
    # (includes the 2px orange frame around each thumbnail)
    GRAVEYARD_CAPTION_HEIGHT = 36

    def graveyard_thumb_height(self):
        """Thumbnail height that fits fully inside the graveyard canvas."""

        canvas_h = self.graveyard_canvas.winfo_height()
        if canvas_h <= 1:
            return self.GRAVEYARD_THUMB_HEIGHT
        return max(
            20,
            min(
                self.GRAVEYARD_THUMB_HEIGHT,
                canvas_h - self.GRAVEYARD_CAPTION_HEIGHT
            )
        )

    def on_graveyard_resize(self, event=None):
        """Re-fit thumbnails when the graveyard's height changes (debounced)."""

        self.center_graveyard()

        if self._graveyard_thumb_h == self.graveyard_thumb_height():
            return
        if self._graveyard_resize_job is not None:
            self.root.after_cancel(self._graveyard_resize_job)
        self._graveyard_resize_job = self.root.after(
            60, self._apply_graveyard_resize
        )

    def _apply_graveyard_resize(self):

        self._graveyard_resize_job = None
        if self.game_snapshots:
            self.refresh_graveyard()

    def center_graveyard(self, event=None):
        """Keep the thumbnails centred; scroll only when they overflow."""

        canvas_w = self.graveyard_canvas.winfo_width()
        inner_w = self.graveyard_inner.winfo_reqwidth()
        inner_h = self.graveyard_inner.winfo_reqheight()
        x = max(0, (canvas_w - inner_w) // 2)
        self.graveyard_canvas.coords(self.graveyard_window, x, 0)
        self.graveyard_canvas.configure(
            scrollregion=(0, 0, max(canvas_w, inner_w), inner_h)
        )

    def refresh_graveyard(self):
        """Rebuild the graveyard thumbnails from this game's snapshots."""

        for child in self.graveyard_inner.winfo_children():
            child.destroy()
        self.graveyard_photos = []

        if not self.game_snapshots:
            tk.Label(
                self.graveyard_inner,
                text="No snapshots this game.",
                bg=BG, fg=DISABLED,
                font=(self._results_serif, 12)
            ).pack(padx=5, pady=5)
            self.graveyard_canvas.xview_moveto(0)
            return

        h = self.graveyard_thumb_height()
        self._graveyard_thumb_h = h

        for n, snapshot in enumerate(self.game_snapshots, start=1):
            image = Image.fromarray(
                cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
            )
            if MIRROR_CAMERA:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            w = max(1, int(image.width * h / image.height))
            image = image.resize((w, h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.graveyard_photos.append(photo)

            cell = tk.Frame(self.graveyard_inner, bg=BG)
            cell.pack(side="left", padx=6)
            tk.Label(
                cell, image=photo, bg=BG, bd=0,
                highlightthickness=2, highlightbackground=ORANGE
            ).pack()
            players = self.game_snapshot_players[n - 1]
            caption = (
                ", ".join(self.player_name(i) for i in players)
                if players else f"#{n}"
            )
            tk.Label(
                cell, text=caption, bg=BG, fg=CREAM,
                font=(self._results_serif, 12)
            ).pack(pady=(4, 0))

        self.graveyard_canvas.xview_moveto(0)

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
        self.refresh_graveyard()
        self.results_frame.pack(fill="both", expand=True)

    # =========================================
    # PLAYER LIST
    # =========================================

    def make_player_row(self, parent, font_size, horizontal=False,
                        themed=False):
        """A player name, a status label and a red strike line.

        With horizontal=True the entries sit side by side in a line.
        With themed=True the row uses the dark "Nobody's Watching" palette.
        """

        if themed:
            row = tk.Frame(parent, bg=BG)
        else:
            row = ttk.Frame(parent)
        if horizontal:
            row.pack(side="left", padx=12, pady=2)
        else:
            row.pack(anchor="w", fill="x", pady=4)

        if themed:
            name_label = tk.Label(
                row, bg=BG, fg=CREAM,
                font=(self._results_serif, font_size + 1)
            )
            status_label = tk.Label(
                row, bg=BG, font=(self._results_serif, font_size - 1, "bold")
            )
            # (survived colour, eliminated colour)
            status_label.status_colors = (SURVIVED, RED)
            strike = tk.Frame(row, bg=RED, height=2)
        else:
            name_label = tk.Label(row, font=("Segoe UI", font_size))
            status_label = ttk.Label(
                row,
                font=("Segoe UI", font_size, "bold")
            )
            status_label.status_colors = ("green", "red")
            # Single red line drawn through the name when eliminated
            strike = tk.Frame(row, bg="red", height=2)

        name_label.pack(side="left")
        status_label.pack(side="left", padx=(10, 0))

        return name_label, status_label, strike

    def player_name(self, index):

        name = self.name_vars[index].get().strip()
        return name or f"Player {index + 1}"

    def refresh_rows(self, rows):

        for i, (name_label, status_label, strike) in enumerate(rows):
            name_label.config(text=f"{i + 1}. {self.player_name(i)}")

            survived_color, eliminated_color = status_label.status_colors

            if self.eliminated[i]:
                status_label.config(text="Eliminated",
                                    foreground=eliminated_color)
                strike.place(
                    in_=name_label,
                    relx=0, rely=0.5, relwidth=1,
                    height=2, anchor="w"
                )
                strike.lift()
            elif self.survived[i]:
                status_label.config(text="Survived",
                                    foreground=survived_color)
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

        # Nobody can survive or be eliminated before the game starts
        if self.in_lobby:
            return

        index = int(event.char) - 1

        if not 0 <= index < self.num_players:
            return

        # Pressing again un-marks, in case of a mistake
        if self.snapshot_showing:
            self.eliminated[index] = not self.eliminated[index]
            if self.eliminated[index]:
                self.survived[index] = False
                # Remember which snapshot caught this player
                if self.current_snapshot is not None:
                    self.game_snapshot_players[self.current_snapshot].append(
                        index
                    )
            else:
                for players in self.game_snapshot_players:
                    if index in players:
                        players.remove(index)
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
            self.display_frame(
                self.latest_frame,
                None if self.snapshot_showing else self.latest_overlay,
                snapshot=self.snapshot_showing
            )

    def display_frame(self, frame, overlay=None, snapshot=False):

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
        frame_w, frame_h = image.size

        # Mirror so players' left/right match what they see on screen.
        # Detection runs on the unmirrored frame, so flip the overlay too.
        if MIRROR_CAMERA:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            overlay = mirror_overlay(overlay, frame_w)

        # Stretch image to fill entire camera widget
        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

        self.draw_camera_overlay(
            image, overlay, width / frame_w, height / frame_h, snapshot
        )

        photo = ImageTk.PhotoImage(image)

        self.camera_label.configure(
            image=photo,
            anchor="center"
        )

        # Keep reference to prevent garbage collection
        self.camera_photo = photo

    # =========================================
    # CAMERA OVERLAY (Figma "Step into the frame" style)
    # =========================================

    def slot_for_track(self, track_id, now):
        """Stable P1/P2/... slot for a tracker ID while it stays visible."""

        # Forget tracks that have not been seen for a while
        for tid in [t for t, (_, seen) in self.track_slots.items()
                    if now - seen > 2.0]:
            del self.track_slots[tid]

        if track_id in self.track_slots:
            slot = self.track_slots[track_id][0]
        else:
            used = {s for s, _ in self.track_slots.values()}
            slot = next(i for i in range(len(used) + 1) if i not in used)
        self.track_slots[track_id] = [slot, now]
        return slot

    def draw_camera_overlay(self, image, overlay, sx, sy, snapshot):
        """Draw title, ROI, player outlines and stats onto the camera image."""

        w, h = image.size
        draw = ImageDraw.Draw(image)

        playing = bool(overlay and overlay.get("playing", True))

        # Lobby: title at the top. Live game: no title, so nothing covers
        # the players. Snapshot: "Caught moving" at the bottom centre so the
        # caught player's face stays visible.
        show_title = snapshot or not playing
        title_at_bottom = snapshot

        # Dark fade behind the title so it reads over the camera
        # (drawn first so outlines and labels stay bright on top of it)
        # (taller in the lobby, where the shut eye sits above the title)
        lobby = show_title and not snapshot
        shade_h = max(1, int(h * (0.40 if lobby else 0.32)))
        if not show_title:
            pass
        elif title_at_bottom:
            image.paste(
                BG, (0, h - shade_h, w, h),
                top_shade_mask(w, shade_h).transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )
            )
        else:
            image.paste(BG, (0, 0, w, shade_h), top_shade_mask(w, shade_h))

        if overlay and not snapshot:

            # Dim everything outside the eye's detection zone
            # (no zone in the lobby: the eye isn't watching yet)
            if overlay.get("roi"):
                rx1, _, rx2, _ = overlay["roi"]
                rx1, rx2 = int(rx1 * sx), int(rx2 * sx)
                for box in ((0, 0, rx1, h), (rx2, 0, w, h)):
                    if box[2] - box[0] > 0:
                        region = image.crop(box)
                        image.paste(
                            Image.blend(region,
                                        Image.new("RGB", region.size, BG),
                                        0.55),
                            box
                        )
                draw.line((rx1, 0, rx1, h), fill=ORANGE, width=2)
                draw.line((rx2, 0, rx2, h), fill=ORANGE, width=2)

            # Player outlines with a pill label above each one
            now = time.monotonic()
            pill_font = load_font(tuple(BOLD_FONT_FILES),
                                  max(14, int(h * 0.026)))
            for track_id, (x1, y1, x2, y2), state in overlay["people"]:
                slot = self.slot_for_track(track_id, now)
                color = PLAYER_COLORS[slot % len(PLAYER_COLORS)]

                pad = 8
                box = (int(x1 * sx) - pad, int(y1 * sy) - pad,
                       int(x2 * sx) + pad, int(y2 * sy) + pad)

                if state == "MOVING":
                    line_color, text = RED, f"P{slot + 1} MOVING"
                elif state == "STATIONARY":
                    # "READY" only makes sense in the lobby
                    line_color = color
                    text = f"P{slot + 1}" if playing else f"P{slot + 1} READY"
                else:
                    line_color, text = color, f"P{slot + 1}"

                draw_dashed_rounded_rect(
                    draw, box, radius=int(min(w, h) * 0.03),
                    color=line_color, width=4
                )

                left, top, right, bottom = draw.textbbox(
                    (0, 0), text, font=pill_font
                )
                pw = (right - left) + 2 * int(pill_font.size * 1.6)
                ph = (bottom - top) + int(pill_font.size * 1.0)
                cx = (box[0] + box[2]) / 2
                py2 = box[1] - 10
                if py2 - ph < 0:          # no room above: tuck inside
                    py2 = box[1] + 10 + ph
                pill = (cx - pw / 2, py2 - ph, cx + pw / 2, py2)

                if state in ("MOVING", "STATIONARY"):
                    draw.rounded_rectangle(pill, radius=ph / 2,
                                           fill=line_color)
                    text_color = INK
                else:  # still being assessed: outlined pill
                    draw.rounded_rectangle(pill, radius=ph / 2, fill=PANEL,
                                           outline=line_color, width=2)
                    text_color = line_color
                draw_centered_text(draw, cx, pill[1] + (ph - (bottom - top)) / 2,
                                   text, pill_font, text_color)

        # Eye at the top centre: in the game it looks where the motor
        # points; in the lobby it is shut (nobody's watching yet) and the
        # title is pushed down below it.
        title_offset = 0
        if overlay and not snapshot:
            ew = max(90, int(w * 0.14))
            eh = int(ew * 150 / 380)
            ey = int(h * 0.03)
            if playing:
                look = round(motor_angle_to_look(overlay["motor"]), 2)
                eye_img = render_eye_image(ew, eh, look)
            else:
                eye_img = render_eye_image(ew, eh, 0.0, closed=True)
                title_offset = ey + eh - int(h * 0.04)
            image.paste(eye_img, ((w - ew) // 2, ey), eye_img)

        if snapshot:
            title = "CAUGHT MOVING"
            subtitle = (f"Press 1-{self.num_players} to eliminate the player, "
                        "then Continue")
        else:
            title = "STEP INTO THE FRAME"
            subtitle = "Stand inside your outline and hold still"

        if show_title:
            self.draw_overlay_title(draw, w, h, title, subtitle,
                                    title_at_bottom, title_offset)

        # Small grey stats in the top-right corner
        if overlay and not snapshot:
            stats = (f"MOTOR {overlay['motor']:.1f}°   "
                     f"PEOPLE {overlay['count']}/{overlay['max_people']}")
            small = load_font(tuple(BODY_FONT_FILES), max(11, int(h * 0.018)))
            left, top, right, bottom = draw.textbbox((0, 0), stats, font=small)
            draw.text((w - 24 - (right - left), 20 - top), stats,
                      font=small, fill=MUTED)

    def draw_overlay_title(self, draw, w, h, title, subtitle, at_bottom,
                           top_offset=0):
        """Big Cinzel title + subtitle, centred at the top or bottom."""

        title_font = load_font(tuple(TITLE_FONT_FILES),
                               max(24, int(h * 0.075)))
        body_font = load_font(tuple(BODY_FONT_FILES),
                              max(13, int(h * 0.03)))
        gap = int(h * 0.025)
        if at_bottom:
            # Measure first so the subtitle sits just above the bottom edge
            _, t_top, _, t_bottom = draw.textbbox((0, 0), title,
                                                  font=title_font)
            _, s_top, _, s_bottom = draw.textbbox((0, 0), subtitle,
                                                  font=body_font)
            y = (h - int(h * 0.05) - (s_bottom - s_top) - gap
                 - (t_bottom - t_top))
        else:
            y = int(h * 0.06) + top_offset
        y += draw_centered_text(draw, w / 2, y, title, title_font, CREAM)
        draw_centered_text(draw, w / 2, y + gap, subtitle, body_font, CREAM)

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

        # Start in the "Step into the frame" lobby
        self.ready_event.clear()
        self.in_lobby = True
        self.show_game_controls(lobby=True)

        # Fresh game: nobody has survived yet, no old frames
        self.survived = [False] * self.num_players
        self.eliminated = [False] * self.num_players
        self.snapshot_showing = False
        self.game_snapshots = []
        self.game_snapshot_players = []
        self.current_snapshot = None
        self.latest_frame = None
        self.latest_overlay = None
        self.track_slots = {}
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
                self.messages,
                self.ready_event
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
        self.in_lobby = False

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

            frame, overlay = self.frame_queue.get_nowait()

            # While a snapshot is on screen, keep it there
            if not self.snapshot_showing:
                self.latest_frame = frame
                self.latest_overlay = overlay
                self.display_frame(frame, overlay)

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

                    # Keep it for the graveyard (ignore stragglers that
                    # arrive after the game was stopped)
                    if self.game_running and not self.stop_event.is_set():
                        self.game_snapshots.append(snapshot.copy())
                        self.game_snapshot_players.append([])
                        self.current_snapshot = len(self.game_snapshots) - 1
                    else:
                        self.current_snapshot = None

                    self.latest_frame = snapshot
                    self.snapshot_showing = True

                    # Display snapshot in centre
                    self.display_frame(snapshot, snapshot=True)

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

                # Homing only needs to happen once per session
                self.home_button.config(
                    state="disabled" if self.motor_homed else "normal"
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
