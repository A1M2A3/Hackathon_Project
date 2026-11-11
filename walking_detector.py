
from collections import defaultdict, deque
import math
import numpy as np


LEFT_HIP = 11
RIGHT_HIP = 12


class WalkingDetector:

    def __init__(self, config):

        self.cfg = config

        # Independent position history for every person
        self.history = defaultdict(lambda: deque())

        # Current state of each person
        self.states = {}

        # Consecutive movement confirmation time
        self.movement_start = {}

        # Last time a person was observed
        self.last_seen = {}

    def update(self, track_id, points, keypoint_conf,
               bbox, timestamp):

        x1, y1, x2, y2 = bbox

        height = max(1.0, float(y2 - y1))

        points = np.asarray(points, dtype=float)
        conf = np.asarray(keypoint_conf, dtype=float)

        # Bounding-box centre
        bbox_center = (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

        # Calculate hip centre if both hips are reliable
        hip_valid = (
            len(conf) > RIGHT_HIP
            and conf[LEFT_HIP] >= self.cfg["min_keypoint_confidence"]
            and conf[RIGHT_HIP] >= self.cfg["min_keypoint_confidence"]
        )

        if hip_valid:

            hip_center = (
                (points[LEFT_HIP, 0] +
                 points[RIGHT_HIP, 0]) / 2,

                (points[LEFT_HIP, 1] +
                 points[RIGHT_HIP, 1]) / 2
            )

        else:
            hip_center = None

        # Use hip centre when available, otherwise bbox centre
        position = hip_center if hip_center is not None else bbox_center

        # Store normalized position
        normalized_position = (
            position[0] / height,
            position[1] / height
        )

        history = self.history[track_id]

        history.append({
            "time": timestamp,
            "position": normalized_position,
            "height": height,
            "hip_valid": hip_center is not None
        })

        self.last_seen[track_id] = timestamp

        # Maintain rolling history
        window = self.cfg["history_seconds"]

        while history and timestamp - history[0]["time"] > window:
            history.popleft()

        # Need sufficient time between observations
        if len(history) < 2:
            self.states.setdefault(track_id, "UNCERTAIN")
            return {"state": self.states[track_id], "displacement": 0.0}

        first = history[0]
        last = history[-1]

        elapsed = last["time"] - first["time"]

        if elapsed < self.cfg["minimum_comparison_seconds"]:
            self.states.setdefault(track_id, "UNCERTAIN")
            return {"state": self.states[track_id], "displacement": 0.0}

        # Calculate normalized displacement
        p1 = first["position"]
        p2 = last["position"]

        displacement = math.dist(p1, p2)

        # Determine movement using hysteresis
        was_moving = self.states.get(track_id) == "MOVING"

        if was_moving:
            threshold = self.cfg["stop_threshold"]
        else:
            threshold = self.cfg["movement_threshold"]

        moving_candidate = displacement >= threshold

        if moving_candidate:

            self.movement_start.setdefault(track_id, timestamp)

            duration = timestamp - self.movement_start[track_id]

            if duration >= self.cfg["confirmation_seconds"]:
                self.states[track_id] = "MOVING"

        else:

            self.movement_start.pop(track_id, None)

            if displacement < self.cfg["stop_threshold"]:
                self.states[track_id] = "STATIONARY"

        self.states.setdefault(track_id, "UNCERTAIN")

        return {
            "state": self.states[track_id],
            "displacement": displacement
        }

    def cleanup(self, timestamp):

        # Remove IDs that have disappeared for too long
        timeout = self.cfg["lost_track_seconds"]

        expired = [
            track_id
            for track_id, last_seen in self.last_seen.items()
            if timestamp - last_seen > timeout
        ]

        for track_id in expired:

            self.history.pop(track_id, None)
            self.states.pop(track_id, None)
            self.movement_start.pop(track_id, None)
            self.last_seen.pop(track_id, None)

