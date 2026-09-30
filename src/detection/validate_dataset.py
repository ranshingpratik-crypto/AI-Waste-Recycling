from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data" / "final_dataset"

SPLITS = ["train", "val", "test"]

errors = []
total_images = 0
total_labels = 0
total_objects = 0

print("=" * 50)
print("FINAL DATASET VALIDATION")
print("=" * 50)

for split in SPLITS:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = list(image_dir.glob("*"))
    images = [
        x for x in images
        if x.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split.upper()}")
    print("-" * 30)
    print("Images:", len(images))
    print("Labels:", len(labels))

    total_images += len(images)
    total_labels += len(labels)

    image_stems = {x.stem for x in images}
    label_stems = {x.stem for x in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    if missing_labels:
        errors.append(
            f"{split}: {len(missing_labels)} images have no labels"
        )

    if missing_images:
        errors.append(
            f"{split}: {len(missing_images)} labels have no images"
        )

    # Validate images
    for image_file in images:

        try:
            with Image.open(image_file) as img:
                img.verify()
        except Exception as e:
            errors.append(
                f"Corrupt image: {image_file.name} ({e})"
            )

    # Validate labels
    for label_file in labels:

        with open(label_file, "r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, 1):

                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 5:
                    errors.append(
                        f"Invalid label format: {label_file.name}, "
                        f"line {line_number}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    errors.append(
                        f"Invalid numbers: {label_file.name}, "
                        f"line {line_number}"
                    )
                    continue

                total_objects += 1

                if not 0 <= class_id <= 6:
                    errors.append(
                        f"Invalid class ID {class_id}: "
                        f"{label_file.name}"
                    )

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    errors.append(
                        f"Invalid bbox: {label_file.name}, "
                        f"line {line_number}"
                    )

print("\n" + "=" * 50)
print("VALIDATION SUMMARY")
print("=" * 50)

print("Total images :", total_images)
print("Total labels :", total_labels)
print("Total objects:", total_objects)

if errors:

    print("\nPROBLEMS FOUND:", len(errors))

    for error in errors[:30]:
        print("-", error)

    if len(errors) > 30:
        print(
            f"... and {len(errors) - 30} more problems."
        )

else:

    print("\n✓ No validation errors found.")
    print("✓ Images and labels are matched.")
    print("✓ Bounding boxes are valid.")
    print("✓ Class IDs are valid.")

print("\nDataset:")
print(DATASET)