from pathlib import Path
import shutil
import random

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TACO_LABELS = PROJECT_ROOT / "data" / "waste_dataset" / "labels"
TACO_IMAGES = PROJECT_ROOT / "data" / "waste_dataset" / "images"

SYN_ROOT = (
    PROJECT_ROOT
    / "external_datasets"
    / "outdoor_waste"
    / "A synthetic outdoor waste image dataset with YOLO-"
    / "SyntheticOutdoorWaste_YOLO"
    / "SynWasteNet"
)

SYN_LABELS = SYN_ROOT / "labels"
SYN_IMAGES = SYN_ROOT / "images"

OUTPUT = PROJECT_ROOT / "data" / "final_dataset"

# ============================================================
# FINAL CLASSES
# ============================================================

CLASSES = [
    "plastic",
    "metal",
    "paper",
    "cardboard",
    "glass",
    "e_waste",
    "organic",
]

# SynWasteNet → final classes
SYN_MAPPING = {
    0: 0,   # plastic → plastic
    1: 2,   # paper → paper
    2: 3,   # cardboard → cardboard
    3: 1,   # metal → metal
    4: 4,   # glass → glass
    5: 6,   # organic → organic
    6: 5,   # battery → e_waste
    7: 5,   # e_waste → e_waste
}

# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ============================================================
# COLLECT DATA
# ============================================================

items = []

# ------------------------------------------------------------
# TACO
# ------------------------------------------------------------

print("\nCollecting TACO dataset...")

for label_file in TACO_LABELS.rglob("*.txt"):

    image_file = None

    for ext in [".jpg", ".jpeg", ".png"]:
        matches = list(TACO_IMAGES.rglob(label_file.stem + ext))

        if matches:
            image_file = matches[0]
            break

    if image_file:
        items.append(("taco", image_file, label_file))

print("TACO samples:", len(items))

# ------------------------------------------------------------
# SYNWASTENET
# ------------------------------------------------------------

print("\nCollecting SynWasteNet dataset...")

syn_items = []

for label_file in SYN_LABELS.rglob("*.txt"):

    valid_lines = []

    with open(label_file, "r", encoding="utf-8") as f:

        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            if class_id in SYN_MAPPING:
                valid_lines.append(line)

    # Skip files containing only cloth/other_waste
    if not valid_lines:
        continue

    image_file = None

    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = SYN_IMAGES / (label_file.stem + ext)

        if candidate.exists():
            image_file = candidate
            break

    if image_file:
        syn_items.append(("syn", image_file, label_file))

items.extend(syn_items)

print("SynWasteNet usable samples:", len(syn_items))
print("Total samples:", len(items))

# ============================================================
# SHUFFLE
# ============================================================

random.seed(42)
random.shuffle(items)

# ============================================================
# SPLIT
# ============================================================

total = len(items)

train_end = int(total * 0.70)
val_end = int(total * 0.90)

splits = {
    "train": items[:train_end],
    "val": items[train_end:val_end],
    "test": items[val_end:],
}

print("\nDataset split:")
for name, data in splits.items():
    print(f"{name}: {len(data)}")

# ============================================================
# COPY + CONVERT
# ============================================================

counters = {
    "plastic": 0,
    "metal": 0,
    "paper": 0,
    "cardboard": 0,
    "glass": 0,
    "e_waste": 0,
    "organic": 0,
}

for split_name, split_items in splits.items():

    for index, (source, image_file, label_file) in enumerate(split_items):

        # Unique filename prevents collisions
        new_name = f"{source}_{index:06d}"

        new_image = OUTPUT / "images" / split_name / f"{new_name}.jpg"
        new_label = OUTPUT / "labels" / split_name / f"{new_name}.txt"

        shutil.copy2(image_file, new_image)

        output_lines = []

        with open(label_file, "r", encoding="utf-8") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                old_class = int(parts[0])

                # ------------------------------------------------
                # TACO labels are already in our old 7-class format
                # ------------------------------------------------

                if source == "taco":

                    if old_class < 0 or old_class >= 7:
                        continue

                    new_class = old_class

                # ------------------------------------------------
                # SynWasteNet conversion
                # ------------------------------------------------

                else:

                    if old_class not in SYN_MAPPING:
                        continue

                    new_class = SYN_MAPPING[old_class]

                parts[0] = str(new_class)

                output_lines.append(" ".join(parts))

                counters[CLASSES[new_class]] += 1

        # Write converted label
        with open(new_label, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

# ============================================================
# CREATE data.yaml
# ============================================================

yaml_file = OUTPUT / "data.yaml"

yaml_content = f"""path: {OUTPUT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}

names:
"""

for i, name in enumerate(CLASSES):
    yaml_content += f"  {i}: {name}\n"

with open(yaml_file, "w", encoding="utf-8") as f:
    f.write(yaml_content)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n========================================")
print("DATASET MERGING COMPLETED")
print("========================================")

print("\nFinal class distribution:")

for name, count in counters.items():
    print(f"{name:12s}: {count}")

print("\nDataset location:")
print(OUTPUT)

print("\nYAML file:")
print(yaml_file)

print("\nFinal classes:")
for i, name in enumerate(CLASSES):
    print(f"{i}: {name}")