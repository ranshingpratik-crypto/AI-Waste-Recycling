from pathlib import Path
import shutil

SOURCE = Path("external_datasets/garbage_detection")
OUTPUT = Path("data/garbage_dataset")

# Original dataset -> our project classes
# 0 BIODEGRADABLE -> 5 organic
# 1 CARDBOARD     -> 3 cardboard
# 2 GLASS         -> 4 glass
# 3 METAL         -> 1 metal
# 4 PAPER         -> 2 paper
# 5 PLASTIC       -> 0 plastic

CLASS_MAP = {
    0: 5,
    1: 3,
    2: 4,
    3: 1,
    4: 2,
    5: 0,
}

SPLITS = ["train", "valid", "test"]

for split in SPLITS:

    src_images = SOURCE / split / "images"
    src_labels = SOURCE / split / "labels"

    out_images = OUTPUT / split / "images"
    out_labels = OUTPUT / split / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = list(src_images.glob("*"))

    converted = 0

    for image_path in image_files:

        label_path = src_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        # Copy image
        shutil.copy2(
            image_path,
            out_images / image_path.name
        )

        new_lines = []

        with open(label_path, "r") as f:
            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                old_class = int(parts[0])

                if old_class not in CLASS_MAP:
                    continue

                new_class = CLASS_MAP[old_class]

                new_lines.append(
                    f"{new_class} "
                    + " ".join(parts[1:])
                )

        # Save converted labels
        with open(
            out_labels / label_path.name,
            "w"
        ) as f:
            f.write("\n".join(new_lines))

        converted += 1

    print(f"{split}: converted {converted} images")

print("\nConversion completed.")
print(f"Output: {OUTPUT}")