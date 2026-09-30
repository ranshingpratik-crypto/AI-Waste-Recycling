from pathlib import Path
import shutil
import random

# Paths
SOURCE_IMAGES = Path("data/final_dataset/test/images")
SOURCE_LABELS = Path("data/final_dataset/test/labels")

OUTPUT_IMAGES = Path("data/real_world/images/test")
OUTPUT_LABELS = Path("data/real_world/labels/test")

# Project classes
CLASSES = {
    0: "plastic",
    1: "metal",
    2: "paper",
    3: "cardboard",
    4: "glass",
    5: "organic",
}

IMAGES_PER_CLASS = 10

random.seed(42)

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

# Find images containing each class
class_images = {class_id: [] for class_id in CLASSES}

for label_file in SOURCE_LABELS.glob("*.txt"):
    lines = label_file.read_text().splitlines()

    present_classes = set()

    for line in lines:
        parts = line.split()

        if not parts:
            continue

        class_id = int(parts[0])

        if class_id in CLASSES:
            present_classes.add(class_id)

    image_file = SOURCE_IMAGES / (label_file.stem + ".jpg")

    # Also check common image extensions
    if not image_file.exists():
        for ext in [".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = SOURCE_IMAGES / (label_file.stem + ext)
            if candidate.exists():
                image_file = candidate
                break

    if not image_file.exists():
        continue

    for class_id in present_classes:
        class_images[class_id].append((image_file, label_file))


# Select images
selected = set()

for class_id, class_name in CLASSES.items():

    candidates = class_images[class_id]

    random.shuffle(candidates)

    count = 0

    for image_file, label_file in candidates:

        if image_file.name in selected:
            continue

        selected.add(image_file.name)

        shutil.copy2(
            image_file,
            OUTPUT_IMAGES / image_file.name
        )

        shutil.copy2(
            label_file,
            OUTPUT_LABELS / label_file.name
        )

        count += 1

        if count >= IMAGES_PER_CLASS:
            break

    print(f"{class_name}: {count} images selected")


print("\nDone!")
print(f"Images saved to: {OUTPUT_IMAGES}")
print(f"Labels saved to: {OUTPUT_LABELS}")
print(f"Total unique images: {len(selected)}")