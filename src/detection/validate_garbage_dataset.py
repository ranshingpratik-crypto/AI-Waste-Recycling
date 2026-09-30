from pathlib import Path
from PIL import Image

BASE = Path("data/final_dataset")

CLASS_NAMES = [
    "plastic",
    "metal",
    "paper",
    "cardboard",
    "glass",
    "organic"
]

SPLITS = ["train", "valid", "test"]

print("=" * 50)
print("GARBAGE DATASET VALIDATION")
print("=" * 50)

total_images = 0
total_objects = 0
total_errors = 0

for split in SPLITS:

    image_dir = BASE / split / "images"
    label_dir = BASE / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    image_stems = {x.stem for x in images}
    label_stems = {x.stem for x in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    class_counts = {i: 0 for i in range(6)}

    split_errors = 0
    objects = 0

    for image_path in images:

        # Check image
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            print(f"Invalid image: {image_path}")
            split_errors += 1

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line_no, line in enumerate(lines, 1):

            parts = line.strip().split()

            if len(parts) != 5:
                print(
                    f"Invalid label format: "
                    f"{label_path} line {line_no}"
                )
                split_errors += 1
                continue

            try:
                class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]
            except ValueError:
                print(
                    f"Invalid values: "
                    f"{label_path} line {line_no}"
                )
                split_errors += 1
                continue

            if class_id not in range(6):
                print(
                    f"Invalid class {class_id}: "
                    f"{label_path}"
                )
                split_errors += 1
                continue

            if not all(0 <= x <= 1 for x in values):
                print(
                    f"Bounding box outside 0-1: "
                    f"{label_path} line {line_no}"
                )
                split_errors += 1
                continue

            class_counts[class_id] += 1
            objects += 1

    if missing_labels:
        print(f"Missing labels: {len(missing_labels)}")
        split_errors += len(missing_labels)

    if missing_images:
        print(f"Missing images: {len(missing_images)}")
        split_errors += len(missing_images)

    print()
    print(split.upper())
    print("-" * 30)
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Objects: {objects}")

    print("Classes:")
    for class_id, count in class_counts.items():
        print(f"  {class_id} {CLASS_NAMES[class_id]}: {count}")

    print(f"Errors: {split_errors}")

    total_images += len(images)
    total_objects += objects
    total_errors += split_errors

print()
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Total images: {total_images}")
print(f"Total objects: {total_objects}")
print(f"Total errors: {total_errors}")

if total_errors == 0:
    print()
    print("DATASET VALIDATION PASSED")
else:
    print()
    print("DATASET VALIDATION FOUND ERRORS")