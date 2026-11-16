import cv2                          # detects players, and creates a snapshot when a player is caught moving
import yaml
import time
from ultralytics import YOLO
import random
import queue
from pytrinamic.connections import UsbTmclInterface

from walking_detector import WalkingDetector    

from motor_test import Motor, COM_PORT, microsteps_per_deg

COM_PORT = "COM5"
microsteps_per_deg = 256*(200/360)

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

def get_detection_roi(position_deg, frame_width, frame_height):

    # Motor operating range
    ANGLE_MIN = 0
    ANGLE_MAX = 90

    # Width of detection region as a fraction of camera width
    ROI_WIDTH_FRACTION = 0.35

    roi_width = int(frame_width * ROI_WIDTH_FRACTION)
    half_width = roi_width // 2

    # Convert motor angle into a normalized value between 0 and 1
    fraction = (position_deg - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
    fraction = max(0, min(1, fraction))

    # Calculate horizontal center of ROI
    center_x = int(
        half_width + fraction * (frame_width - 2 * half_width)
    )

    x1 = center_x - half_width
    x2 = center_x + half_width

    return x1, 0, x2, frame_height


def main(motor, stop_event, continue_event, frame_queue, messages):
    # TMCL command opcodes
    MST = 3
    MVP = 4
    SAP = 5
    GAP = 6
    RFS = 13
    
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg["model"]["weights"])
    walker = WalkingDetector(cfg["walking"])
    previous_states = {}

    cap = cv2.VideoCapture(cfg["camera"]["source"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["camera"]["fps_target"])

    if not cap.isOpened():
        messages.put("ERROR: Could not open camera.")
        return

    messages.put("Cyclops running.")

    motor_position = 0
    last_motor_read = 0
    motor_stopped = False

    def send_frame(frame):
        """Keep only the newest live camera frame in the queue."""
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    try:
        while not stop_event.is_set():

            # Check whether the motor has reached its target
            response = motor.send_command(6, 8, 0)

            if response.is_valid:
                if response.value == 1 or motor_stopped:

                    motor_stopped = False
                    new_pos = random.randint(10, 80)

                    motor.send_command(
                        MVP, 0, motor.angle_to_steps(new_pos)
                    )

            # Read camera frame
            ok, frame = cap.read()

            if not ok:
                messages.put("ERROR: Could not read camera frame.")
                break

            now = time.monotonic()

            # Read motor position every 100 ms
            if now - last_motor_read >= 0.1:
                motor_position = motor.get_position()
                last_motor_read = now

            h, w = frame.shape[:2]

            roi_x1, roi_y1, roi_x2, roi_y2 = get_detection_roi(
                motor_position, w, h
            )

            # Run YOLO tracking
            results = model.track(
                frame,
                persist=True,
                tracker=cfg["tracking"]["tracker"],
                classes=[0],
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

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Only process people inside the ROI
                        if not (
                            roi_x1 <= center_x <= roi_x2
                            and roi_y1 <= center_y <= roi_y2
                        ):
                            continue

                        pose = keypoints[i]
                        points = pose[:, :2]

                        kp_conf = (
                            pose[:, 2]
                            if pose.shape[1] > 2
                            else [1.0] * len(pose)
                        )

                        state = walker.update(
                            track_id=track_id,
                            points=points,
                            keypoint_conf=kp_conf,
                            bbox=(x1, y1, x2, y2),
                            timestamp=now
                        )

                        motion_state = state["state"]

                        # Detect transition into MOVING
                        if (
                            motion_state == "MOVING"
                            and previous_states.get(track_id) != "MOVING"
                        ):

                            # Crop the person from the original frame
                            crop_x1 = max(0, min(w, int(x1)))
                            crop_x2 = max(0, min(w, int(x2)))
                            crop_y1 = max(0, min(h, int(y1)))
                            crop_y2 = max(0, min(h, int(y2)))

                            if (
                                crop_x2 > crop_x1
                                and crop_y2 > crop_y1
                            ):

                                snapshot = frame[
                                    crop_y1:crop_y2,
                                    crop_x1:crop_x2
                                ].copy()

                                # Send snapshot to Tkinter
                                messages.put(("snapshot", snapshot))

                            messages.put(
                                f"Person {track_id} detected! "
                                "Waiting for Continue."
                            )

                            # Stop motor
                            motor.send_command(MST, 0, 0)
                            motor_stopped = True

                            # Pause until Continue or Stop is clicked
                            continue_event.clear()

                            while (
                                not continue_event.is_set()
                                and not stop_event.is_set()
                            ):
                                time.sleep(0.05)

                            if stop_event.is_set():
                                break

                            continue_event.clear()
                            messages.put("Game resumed.")

                        previous_states[track_id] = motion_state

                        # Draw bounding box
                        if motion_state == "MOVING":
                            color = (0, 0, 255)
                        elif motion_state == "STATIONARY":
                            color = (0, 200, 0)
                        else:
                            color = (0, 255, 255)

                        label = f"ID {track_id}: {motion_state}"

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                        if cfg["display"]["show_skeleton"]:
                            draw_pose(
                                frame,
                                points,
                                kp_conf,
                                cfg["detection"]["keypoint_confidence"]
                            )

                    if stop_event.is_set():
                        break

            # Clean up old tracking information
            walker.cleanup(now)

            # Draw ROI
            cv2.rectangle(
                frame,
                (roi_x1, roi_y1),
                (roi_x2, roi_y2),
                (255, 0, 0),
                2
            )

            # Display motor position
            cv2.putText(
                frame,
                f"Motor: {motor_position:.1f} degrees",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Display number of tracked people
            people_count = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            cv2.putText(
                frame,
                f"People tracked: {people_count}/"
                f"{cfg['camera']['max_people']}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Send live feed to Tkinter
            send_frame(frame)

    except Exception as e:
        messages.put(f"ERROR: {e}")
        raise

    finally:
        # Stop motor
        try:
            motor.send_command(MST, 0, 0)
        except Exception:
            pass

        cap.release()
        messages.put("Game stopped.")


if __name__ == "__main__":
    main()