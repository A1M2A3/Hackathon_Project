import cv2                          # detects players, and creates a snapshot when a player is caught moving
import yaml
import time
from ultralytics import YOLO

from walking_detector import WalkingDetector    

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_pose(frame, points, confidence, threshold=0.35):
    """Draw COCO-17 keypoints and limb connections."""
    for a, b in SKELETON:
        if confidence[a] >= threshold and confidence[b] >= threshold:
            p1 = tuple(points[a].astype(int))
            p2 = tuple(points[b].astype(int))
            cv2.line(frame, p1, p2, (255, 180, 0), 2)

    for i, point in enumerate(points):
        if confidence[i] >= threshold:
            cv2.circle(frame, tuple(point.astype(int)), 3, (255, 255, 255), -1)

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg["model"]["weights"])
    walker = WalkingDetector(cfg["walking"])

    snapshots = {}
    snapshot_until = {}
    previous_states = {}

    cap = cv2.VideoCapture(cfg["camera"]["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["camera"]["fps_target"])

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check camera.source in config.yaml.")

    print("Cyclops running. Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            now = time.monotonic()
            results = model.track(
                frame,
                persist=True,
                tracker=cfg["tracking"]["tracker"],
                classes=[0],                  # COCO class 0 = person
                conf=cfg["detection"]["confidence"],
                imgsz=cfg["detection"]["image_size"],
                max_det=cfg["camera"]["max_people"],
                verbose=False
            )
            result = results[0]

            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes
                keypoints = result.keypoints.data.cpu().numpy()

                if boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    xyxy = boxes.xyxy.cpu().numpy()
                    for i, track_id in enumerate(ids):
                        x1, y1, x2, y2 = xyxy[i].astype(int)
                        box_height = max(1, y2 - y1)
                        pose = keypoints[i]
                        points = pose[:, :2]
                        kp_conf = pose[:, 2] if pose.shape[1] > 2 else [1.0] * len(pose)

                        state = walker.update(
                            track_id=track_id,
                            points=points,
                            keypoint_conf=kp_conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=now
                        )

                        motion_state = state["state"]
                        
                        # Capture only on the transition from STATIONARY to MOVING
                        if (
                            motion_state == "MOVING"
                            and previous_states.get(track_id) != "MOVING"
                            and track_id not in snapshots
                        ):
                            h, w = frame.shape[:2]

                            crop_x1 = max(0, min(w, int(x1)))
                            crop_x2 = max(0, min(w, int(x2)))
                            crop_y1 = max(0, min(h, int(y1)))
                            crop_y2 = max(0, min(h, int(y2)))

                            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                                snapshots[track_id] = frame[
                                    crop_y1:crop_y2, crop_x1:crop_x2
                                ].copy()

                                snapshot_until[track_id] = now + 5.0                     
                            
                        previous_states[track_id] = motion_state

                        if motion_state == "MOVING":
                            color = (0, 0, 255)
                        elif motion_state == "STATIONARY":
                            color = (0, 200, 0)
                        else:
                            color = (0, 255, 255)

                        label = f"ID {track_id}: {motion_state}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        if cfg["display"]["show_skeleton"]:
                            draw_pose(frame, points, kp_conf,
                                      cfg["detection"]["keypoint_confidence"])

            walker.cleanup(now)
            for track_id in list(snapshots.keys()):

                window_name = f"Movement Detected - ID {track_id}"

                if now < snapshot_until[track_id]:
                    cv2.imshow(
                        window_name,
                        snapshots[track_id]
                    )
                else:
                    cv2.destroyWindow(window_name)
                    del snapshots[track_id]
                    del snapshot_until[track_id]
                    
            cv2.putText(frame, f"People tracked: {len(result.boxes) if result.boxes is not None else 0}/"
                               f"{cfg['camera']['max_people']}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Cyclops - Person Walking Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()