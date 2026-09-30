from pathlib import Path
import random
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = PROJECT_ROOT / "data" / "final_dataset"
OUTPUT = PROJECT_ROOT / "results" / "dataset_preview"

CLASSES = [
    "plastic",
    "metal",
    "paper",
    "cardboard",
    "glass",
    "e_waste",
    "organic",
]

OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ------------------------------------------------------------
# Choose samples from TEST dataset
# ------------------------------------------------------------

image_dir = DATASET / "images" / "test"
label_dir = DATASET / "labels" / "test"

images = list(image_dir.glob("*.jpg"))

# Select up to 20 random images
samples = random.sample(images, min(20, len(images)))

print("Creating visual dataset preview...")

for index, image_path in enumerate(samples):

    label_path = label_dir / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    height, width = image.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            label = CLASSES[class_id]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    output_path = OUTPUT / f"sample_{index + 1:02d}.jpg"

    cv2.imwrite(str(output_path), image)

print("\nPreview created successfully.")
print("Location:")
print(OUTPUT)