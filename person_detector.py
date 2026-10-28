
import cv2
from ultralytics import YOLO

# Load a small, pretrained object detection model
model = YOLO("yolo11n.pt")

# Open your laptop webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press Q to quit.")

while True:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Detect people (COCO class 0 = person)
    results = model.predict(
        frame,
        classes=[0],
        conf=0.4,
        imgsz=640,
        verbose=False
    )

    # Draw a bounding box around each detected person
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Person: {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Display the live video
    cv2.imshow("Person Detector", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
