from pathlib import Path
import random
import cv2

ROOT = Path(__file__).resolve().parents[2]

SYN_ROOT = (
    ROOT
    / "external_datasets"
    / "outdoor_waste"
    / "A synthetic outdoor waste image dataset with YOLO-"
    / "SyntheticOutdoorWaste_YOLO"
    / "SynWasteNet"
)

IMAGE_DIR = SYN_ROOT / "split" / "train" / "images"
LABEL_DIR = SYN_ROOT / "split" / "train" / "labels"

CLASSES = [
    "plastic",
    "paper",
    "cardboard",
    "metal",
    "glass",
    "organic",
    "battery",
    "e_waste",
    "cloth",
    "other_waste",
]

OUTPUT = ROOT / "results" / "syn_source_check"
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

images = list(IMAGE_DIR.glob("*.jpg"))

samples = random.sample(images, min(20, len(images)))

print("SynWasteNet source visual check")
print("Images found:", len(images))
print("Creating", len(samples), "samples...")

for i, image_path in enumerate(samples):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    h, w = image.shape[:2]

    if not label_path.exists():
        print("Missing label:", image_path.name)
        continue

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x = float(parts[1]) * w
            y = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h

            x1 = max(0, int(x - bw / 2))
            y1 = max(0, int(y - bh / 2))
            x2 = min(w - 1, int(x + bw / 2))
            y2 = min(h - 1, int(y + bh / 2))

            name = (
                CLASSES[class_id]
                if 0 <= class_id < len(CLASSES)
                else f"UNKNOWN_{class_id}"
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                name,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imwrite(
        str(OUTPUT / f"source_{i+1:02d}.jpg"),
        image
    )

print("\nDone.")
print("Preview folder:")
print(OUTPUT)