from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/detect/waste_yolo11n/weights/best.pt"

# Load model only once
model = YOLO(MODEL_PATH)


def detect_frame(frame):
    """
    Detect waste objects in a single image/frame.

    Returns:
        list of dictionaries containing:
        class, confidence, bbox, track_id
    """

    results = model.track(
        source=frame,
        imgsz=640,
        conf=0.50,
        iou=0.5,
        device=0,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is None:
        return detections

    names = model.names

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        track_id = None

        if box.id is not None:
            track_id = int(box.id[0])

        detections.append({
            "class_id": class_id,
            "class": names[class_id],
            "confidence": round(confidence, 3),
            "bbox": [
                round(x1),
                round(y1),
                round(x2),
                round(y2)
            ],
            "track_id": track_id
        })

    return detections