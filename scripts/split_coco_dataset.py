import json
import random
import shutil
from pathlib import Path


# =============================
# CONFIG
# =============================
SOURCE_JSON = Path("datasets/basketball/annotations/instances_default.json")
SOURCE_IMAGES = Path("datasets/basketball/all_images")

TRAIN_DIR = Path("datasets/basketball/train2017")
VAL_DIR = Path("datasets/basketball/val2017")
ANNOTATIONS_DIR = Path("datasets/basketball/annotations")

TRAIN_JSON = ANNOTATIONS_DIR / "instances_train2017.json"
VAL_JSON = ANNOTATIONS_DIR / "instances_val2017.json"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# =============================
# HELPERS
# =============================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_coco_subset(coco_data, selected_images):
    selected_image_ids = {img["id"] for img in selected_images}

    selected_annotations = [
        ann for ann in coco_data["annotations"]
        if ann["image_id"] in selected_image_ids
    ]

    return {
        "licenses": coco_data.get("licenses", []),
        "info": coco_data.get("info", {}),
        "categories": coco_data.get("categories", []),
        "images": selected_images,
        "annotations": selected_annotations,
    }


def copy_images(image_list, source_dir: Path, destination_dir: Path):
    for image in image_list:
        filename = image["file_name"]
        src = source_dir / filename
        dst = destination_dir / filename

        if not src.exists():
            print(f"WARNING: missing image file: {src}")
            continue

        shutil.copy2(src, dst)


# =============================
# MAIN
# =============================
def main():
    random.seed(RANDOM_SEED)

    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(ANNOTATIONS_DIR)

    with open(SOURCE_JSON, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_coco = build_coco_subset(coco_data, train_images)
    val_coco = build_coco_subset(coco_data, val_images)

    with open(TRAIN_JSON, "w") as f:
        json.dump(train_coco, f, indent=2)

    with open(VAL_JSON, "w") as f:
        json.dump(val_coco, f, indent=2)

    copy_images(train_images, SOURCE_IMAGES, TRAIN_DIR)
    copy_images(val_images, SOURCE_IMAGES, VAL_DIR)

    print("Done.")
    print(f"Total images: {len(images)}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")
    print(f"Train annotations: {len(train_coco['annotations'])}")
    print(f"Val annotations: {len(val_coco['annotations'])}")


if __name__ == "__main__":
    main()
