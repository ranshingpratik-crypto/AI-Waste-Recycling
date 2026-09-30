from pathlib import Path
import random
import shutil
from collections import defaultdict

SOURCE = Path("data/garbage_dataset")
OUTPUT = Path("data/balanced_dataset")

CLASS_NAMES = [
    "plastic",
    "metal",
    "paper",
    "cardboard",
    "glass",
    "organic"
]

# Desired split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# Read all images from all existing splits
all_images = []

for split in ["train", "valid", "test"]:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            all_images.append((image_path, label_path))

print(f"Total images found: {len(all_images)}")

# Group images according to the classes they contain
class_images = defaultdict(set)

for image_path, label_path in all_images:

    classes_in_image = set()

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 5:
                class_id = int(parts[0])

                if 0 <= class_id < 6:
                    classes_in_image.add(class_id)

    for class_id in classes_in_image:
        class_images[class_id].add((image_path, label_path))

# Shuffle complete dataset
random.shuffle(all_images)

# Initial split
total = len(all_images)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]

splits = {
    "train": train_images,
    "valid": val_images,
    "test": test_images
}

# Create folders
for split in splits:

    (OUTPUT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )

# Copy files
for split, files in splits.items():

    for image_path, label_path in files:

        shutil.copy2(
            image_path,
            OUTPUT / split / "images" / image_path.name
        )

        shutil.copy2(
            label_path,
            OUTPUT / split / "labels" / label_path.name
        )

    print(f"{split}: {len(files)} images")

print()
print("Balanced split created.")
print(f"Output: {OUTPUT}")