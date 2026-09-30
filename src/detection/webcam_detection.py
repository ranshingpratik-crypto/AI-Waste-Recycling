import cv2
from collections import Counter

from detection_engine import detect_frame


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("AI Waste Detection started.")
print("Press 'q' to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Get structured detections
    detections = detect_frame(frame)

    # Display detections
    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        label = (
            f'{detection["class"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Count detected materials
    class_counts = Counter(
        detection["class"]
        for detection in detections
    )

    # Show summary
    y = 30

    for material, count in class_counts.items():

        text = f"{material}: {count}"

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        y += 35

    cv2.imshow("AI Waste Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()