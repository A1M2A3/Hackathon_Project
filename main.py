
from collections import defaultdict, deque
import time

import cv2
from ultralytics import YOLO

# Pose model performs person detection and joint estimation.
model = YOLO("yolo11n-pose.pt")

# One independent history per tracked person.
pose_history = defaultdict(lambda: deque(maxlen=60))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

previous_time = time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=640,
        conf=0.35,
        verbose=False
    )

    result = results[0]

    if result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        ids = (
            result.boxes.id.int().cpu().tolist()
            if result.boxes is not None
            and result.boxes.id is not None
            else [None] * len(keypoints)
        )

        for person_id, joints in zip(ids, keypoints):
            if person_id is None:
                continue

            pose_history[person_id].append(joints)

            # TODO: classify_action() should analyze the
            # rolling joint history, not a single frame.
            # action, confidence = classify_action(
            #     list(pose_history[person_id])
            # )

    annotated = result.plot()

    now = time.perf_counter()
    fps = 1.0 / max(now - previous_time, 1e-6)
    previous_time = now

    cv2.putText(
        annotated, f"Loop FPS: {fps:.1f}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    cv2.imshow("Person Action Tracker", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



