from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont

# Dataset
BASE = Path("external_datasets/garbage_detection")
IMAGE_DIR = BASE / "train/images"
LABEL_DIR = BASE / "train/labels"

OUTPUT = Path("results/garbage_source_check")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "BIODEGRADABLE",
    "CARDBOARD",
    "GLASS",
    "METAL",
    "PAPER",
    "PLASTIC"
]

# Get images
images = list(IMAGE_DIR.glob("*"))

# Select 20 random images
random.seed(42)
samples = random.sample(images, min(20, len(images)))

for i, image_path in enumerate(samples, 1):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    if label_path.exists():

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # YOLO normalized coordinates → pixels
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            class_name = CLASS_NAMES[class_id]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 18)),
                class_name,
                fill="red"
            )

    output_path = OUTPUT / f"source_{i:02d}.jpg"
    image.save(output_path)

print("Done!")
print(f"Generated {len(samples)} images.")
print(f"Saved to: {OUTPUT}")