from pathlib import Path
import shutil
import random

# --------------------------------------------------
# SOURCE DATASETS
# --------------------------------------------------

GARBAGE = Path("data/balanced_dataset")
TACO = Path("data/waste_dataset")

# Final dataset
OUTPUT = Path("data/final_dataset")

# Final project classes
CLASS_NAMES = [
    "plastic",
    "metal",
    "paper",
    "cardboard",
    "glass",
    "organic"
]

# TACO class mapping
# TACO:
# 0 plastic_bottle -> 0 plastic
# 1 metal_can      -> 1 metal
# 2 paper          -> 2 paper
# 3 cardboard      -> 3 cardboard
# 4 glass          -> 4 glass
#
# TACO 5 and 6 are intentionally excluded.

TACO_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4
}

random.seed(42)


# --------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# --------------------------------------------------

for split in ["train", "valid", "test"]:

    (OUTPUT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# COPY GARBAGE DATASET
# --------------------------------------------------

print("=" * 60)
print("ADDING GARBAGE CLASSIFICATION DATASET")
print("=" * 60)

garbage_counts = {
    "train": 0,
    "valid": 0,
    "test": 0
}

for split in ["train", "valid", "test"]:

    image_dir = GARBAGE / split / "images"
    label_dir = GARBAGE / split / "labels"

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        # Prefix filename to avoid collisions
        new_name = f"garbage_{image_path.name}"

        shutil.copy2(
            image_path,
            OUTPUT / split / "images" / new_name
        )

        shutil.copy2(
            label_path,
            OUTPUT / split / "labels" /
            f"garbage_{label_path.name}"
        )

        garbage_counts[split] += 1

    print(
        f"{split}: {garbage_counts[split]} images"
    )


# --------------------------------------------------
# ADD TACO
# --------------------------------------------------

print()
print("=" * 60)
print("ADDING TACO DATASET")
print("=" * 60)

taco_counts = {
    "train": 0,
    "valid": 0,
    "test": 0
}

# TACO currently has its own train/val/test structure
# We keep the same split.

for split in ["train", "val", "test"]:

    image_dir = TACO / "images" / split
    label_dir = TACO / "labels" / split

    if not image_dir.exists():
        print(f"TACO {split}: folder not found")
        continue

    # Convert TACO split names to our names
    output_split = "valid" if split == "val" else split

    for image_path in image_dir.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        new_lines = []

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                old_class = int(parts[0])

                # Ignore e-waste and organic from TACO
                if old_class not in TACO_MAP:
                    continue

                new_class = TACO_MAP[old_class]

                new_lines.append(
                    f"{new_class} "
                    + " ".join(parts[1:])
                )

        # Don't copy an image if no usable objects remain
        if not new_lines:
            continue

        new_name = f"taco_{image_path.name}"

        shutil.copy2(
            image_path,
            OUTPUT / output_split / "images" / new_name
        )

        with open(
            OUTPUT / output_split / "labels" /
            f"taco_{label_path.name}",
            "w"
        ) as f:

            f.write("\n".join(new_lines))

        taco_counts[output_split] += 1

    print(
        f"{split} -> {output_split}: "
        f"{taco_counts[output_split]} images"
    )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print("=" * 60)
print("FINAL DATASET CREATED")
print("=" * 60)

total = 0

for split in ["train", "valid", "test"]:

    count = garbage_counts[split] + taco_counts[split]

    total += count

    print(
        f"{split}: {count} images "
        f"({garbage_counts[split]} Garbage + "
        f"{taco_counts[split]} TACO)"
    )

print()
print(f"TOTAL IMAGES: {total}")
print(f"Output: {OUTPUT}")

print()
print("FINAL CLASSES:")

for i, name in enumerate(CLASS_NAMES):

    print(f"{i}: {name}")

print()
print("Merge completed successfully.")