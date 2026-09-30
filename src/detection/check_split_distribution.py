from pathlib import Path
from collections import Counter

BASE = Path("data/balanced_dataset")

CLASS_NAMES = [
    "plastic",
    "metal",
    "paper",
    "cardboard",
    "glass",
    "organic"
]

for split in ["train", "valid", "test"]:

    label_dir = BASE / split / "labels"

    counts = Counter()
    images_with_class = Counter()

    label_files = list(label_dir.glob("*.txt"))

    for label_file in label_files:

        classes_in_image = set()

        with open(label_file, "r") as f:
            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                if 0 <= class_id < 6:
                    counts[class_id] += 1
                    classes_in_image.add(class_id)

        for class_id in classes_in_image:
            images_with_class[class_id] += 1

    print()
    print("=" * 45)
    print(split.upper())
    print("=" * 45)

    print(f"Images: {len(label_files)}")
    print()

    for class_id, name in enumerate(CLASS_NAMES):

        print(
            f"{name:<12} "
            f"objects: {counts[class_id]:>6} | "
            f"images: {images_with_class[class_id]:>6}"
        )