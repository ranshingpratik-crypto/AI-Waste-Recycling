import json
import shutil
import random
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_FILE = PROJECT_ROOT / "TACO" / "data" / "annotations.json"
TACO_DATA_DIR = PROJECT_ROOT / "TACO" / "data"

OUTPUT_DIR = PROJECT_ROOT / "data" / "waste_dataset"

RANDOM_SEED = 42

# Train / validation / test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


# ============================================================
# TARGET CLASSES
# ============================================================

CLASS_MAPPING = {
    "plastic_bottle": [
        "Other plastic bottle",
        "Clear plastic bottle",
    ],

    "metal_can": [
        "Drink can",
    ],

    "paper": [
        "Magazine paper",
        "Tissues",
        "Wrapping paper",
        "Normal paper",
        "Paper bag",
        "Paper straw",
    ],

    "cardboard": [
        "Other carton",
        "Egg carton",
        "Drink carton",
        "Corrugated carton",
        "Meal carton",
        "Pizza box",
    ],

    "glass": [
        "Glass bottle",
        "Broken glass",
        "Glass cup",
        "Glass jar",
    ],

    "e_waste": [
        "Battery",
    ],

    "organic_waste": [
        "Food waste",
    ],
}


CLASS_NAMES = list(CLASS_MAPPING.keys())

CLASS_IDS = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_image(filename):
    """
    Search recursively inside TACO/data for the image.
    """

    matches = list(TACO_DATA_DIR.rglob(Path(filename).name))

    if not matches:
        return None

    return matches[0]


def convert_bbox_to_yolo(bbox, image_width, image_height):
    """
    Convert COCO bounding box:
        [x, y, width, height]

    to YOLO:
        [x_center, y_center, width, height]

    All coordinates are normalized to 0-1.
    """

    x, y, width, height = bbox

    x_center = x + width / 2
    y_center = y + height / 2

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # Keep values within valid range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def create_directories():

    for split in ["train", "val", "test"]:

        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

print("=" * 60)
print("TACO → YOLO DATASET CONVERTER")
print("=" * 60)

print("\nLoading annotations...")

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Annotations loaded.")

print("Images in annotation file:", len(data["images"]))
print("Objects/annotations:", len(data["annotations"]))


# ============================================================
# CATEGORY MAPPING
# ============================================================

taco_categories = {
    category["id"]: category["name"]
    for category in data["categories"]
}

category_to_project_class = {}

for project_class, taco_names in CLASS_MAPPING.items():

    for taco_name in taco_names:

        found = False

        for category_id, category_name in taco_categories.items():

            if category_name.lower() == taco_name.lower():

                category_to_project_class[
                    category_id
                ] = project_class

                found = True

                print(
                    f"Mapped: {category_name} "
                    f"→ {project_class}"
                )

                break

        if not found:
            print(
                f"WARNING: Category not found: "
                f"{taco_name}"
            )


# ============================================================
# CREATE IMAGE → ANNOTATION STRUCTURE
# ============================================================

annotations_by_image = {}

for annotation in data["annotations"]:

    image_id = annotation["image_id"]

    category_id = annotation["category_id"]

    if category_id not in category_to_project_class:
        continue

    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []

    annotations_by_image[image_id].append(annotation)


print(
    "\nImages containing our target classes:",
    len(annotations_by_image)
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

create_directories()


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

image_ids = list(annotations_by_image.keys())

random.seed(RANDOM_SEED)
random.shuffle(image_ids)

total = len(image_ids)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_ids = image_ids[:train_end]
val_ids = image_ids[train_end:val_end]
test_ids = image_ids[val_end:]


splits = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids,
}


print("\nDataset split:")
print("Train:", len(train_ids))
print("Validation:", len(val_ids))
print("Test:", len(test_ids))


# ============================================================
# IMAGE LOOKUP
# ============================================================

images_by_id = {
    image["id"]: image
    for image in data["images"]
}


# ============================================================
# CONVERT DATASET
# ============================================================

class_counter = Counter()

copied_images = 0
created_labels = 0

for split_name, split_ids in splits.items():

    print(
        f"\nProcessing {split_name} dataset..."
    )

    for image_id in split_ids:

        image_info = images_by_id[image_id]

        file_name = image_info["file_name"]

        image_width = image_info["width"]
        image_height = image_info["height"]

        source_image = find_image(file_name)

        if source_image is None:

            print(
                "WARNING: Image not found:",
                file_name
            )

            continue

        # ----------------------------------------------------
        # Create unique output filename
        # ----------------------------------------------------

        original_name = Path(file_name).name

        output_name = (
            f"{image_id}_{original_name}"
        )

        destination_image = (
            OUTPUT_DIR
            / "images"
            / split_name
            / output_name
        )

        destination_label = (
            OUTPUT_DIR
            / "labels"
            / split_name
            / f"{Path(output_name).stem}.txt"
        )

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            source_image,
            destination_image
        )

        copied_images += 1

        # ----------------------------------------------------
        # Create YOLO labels
        # ----------------------------------------------------

        label_lines = []

        for annotation in annotations_by_image[image_id]:

            category_id = annotation["category_id"]

            project_class = (
                category_to_project_class[
                    category_id
                ]
            )

            class_id = CLASS_IDS[
                project_class
            ]

            bbox = annotation["bbox"]

            # Ignore invalid bounding boxes
            if len(bbox) != 4:
                continue

            x, y, width, height = bbox

            if width <= 0 or height <= 0:
                continue

            yolo_bbox = convert_bbox_to_yolo(
                bbox,
                image_width,
                image_height
            )

            x_center, y_center, w, h = yolo_bbox

            label_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

            class_counter[project_class] += 1

        if label_lines:

            with open(
                destination_label,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(label_lines)
                )

            created_labels += 1


# ============================================================
# CREATE data.yaml
# ============================================================

yaml_path = OUTPUT_DIR / "data.yaml"

with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"path: {OUTPUT_DIR.as_posix()}\n"
    )

    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")

    f.write("names:\n")

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        f.write(
            f"  {class_id}: {class_name}\n"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("CONVERSION COMPLETED")
print("=" * 60)

print(
    "Images copied:",
    copied_images
)

print(
    "Label files created:",
    created_labels
)

print("\nObject distribution:")

for class_name in CLASS_NAMES:

    print(
        f"{class_name:20s}: "
        f"{class_counter[class_name]}"
    )

print(
    "\nDataset location:"
)

print(OUTPUT_DIR)

print(
    "\ndata.yaml created at:"
)

print(yaml_path)

print("\nDone!")