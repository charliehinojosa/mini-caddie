#!/usr/bin/env python3
"""
Merge multiple Roboflow YOLOv8 datasets into one unified golf dataset.
Remaps class IDs from each dataset to a single unified class list.
"""

import os
import shutil
from pathlib import Path

# ── Unified class names ──────────────────────────────────────────────────────
UNIFIED_CLASSES = [
    "golf_ball",        # 0
    "golf_club",        # 1
    "golf_club_head",   # 2
    "golf_hole",        # 3
    "golf_mat",         # 4
    "person",           # 5
    "player_not_ready", # 6
    "player_ready",     # 7
]

# ── Class mapping: source class name → unified class ID ─────────────────────
CLASS_MAP = {
    # Ball dataset classes → all map to golf_ball (0)
    "Golf-ball": 0,
    "ball": 0,
    "balle blanche": 0,
    "balle jaune": 0,
    "golf": 0,
    "golf ball": 0,
    "golfball": 0,
    # Club dataset classes
    "Golf-club": 1,
    "Golf-club-head": 2,
    # Swing dataset classes
    "club": 1,
    "club_head": 2,
    "golf_hole": 3,
    "golf_mat": 4,
    "person": 5,
    "player_not_ready": 6,
    "player_ready": 7,
}

# ── Dataset configs ──────────────────────────────────────────────────────────
DATASETS = [
    {"name": "ball",  "classes": ["Golf-ball", "ball", "balle blanche", "balle jaune", "golf", "golf ball"]},
    {"name": "club",  "classes": ["Golf-ball", "Golf-club", "Golf-club-head"]},
    {"name": "swing", "classes": ["club", "club_head", "golf_hole", "golf_mat", "golfball", "person", "player_not_ready", "player_ready"]},
]

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "unified"
SPLITS = ["train", "valid", "test"]


def remap_label_file(label_path, source_classes):
    """Read a YOLO label file, remap class IDs, return new content."""
    new_lines = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_id = int(parts[0])
                if old_id >= len(source_classes):
                    continue  # skip invalid
                class_name = source_classes[old_id]
                new_id = CLASS_MAP.get(class_name)
                if new_id is None:
                    continue  # skip unmapped
                new_parts = [str(new_id)] + parts[1:]
                new_lines.append(" ".join(new_parts))
    except Exception as e:
        print(f"  ⚠️ Error reading {label_path}: {e}")
    return "\n".join(new_lines) + ("\n" if new_lines else "")


def main():
    print("⛳ Mini Caddie — Merging Golf Datasets")
    print("=" * 50)

    # Create output structure
    for split in SPLITS:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_labels = 0

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_classes = ds["classes"]
        print(f"\n📦 Processing dataset: {ds_name} ({len(ds_classes)} classes)")

        for split in SPLITS:
            img_dir = BASE_DIR / ds_name / split / "images"
            lbl_dir = BASE_DIR / ds_name / split / "labels"

            if not img_dir.exists():
                print(f"  ⚠️ {split}/images not found, skipping")
                continue

            img_count = 0
            lbl_count = 0

            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                # Prefix with dataset name to avoid collisions
                new_name = f"{ds_name}_{img_file.name}"

                # Copy image
                dst_img = OUTPUT_DIR / split / "images" / new_name
                shutil.copy2(img_file, dst_img)
                img_count += 1

                # Find and remap label
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    new_content = remap_label_file(lbl_file, ds_classes)
                    dst_lbl = OUTPUT_DIR / split / "labels" / (f"{ds_name}_{img_file.stem}.txt")
                    with open(dst_lbl, "w") as f:
                        f.write(new_content)
                    lbl_count += 1

            print(f"  {split}: {img_count} images, {lbl_count} labels")
            total_images += img_count
            total_labels += lbl_count

    # Write unified data.yaml
    yaml_content = f"""# Mini Caddie — Unified Golf Dataset
# Merged from: ball, club, swing datasets

train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(UNIFIED_CLASSES)}
names: {UNIFIED_CLASSES}
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n{'=' * 50}")
    print(f"✅ Merge complete!")
    print(f"   Total images: {total_images}")
    print(f"   Total labels: {total_labels}")
    print(f"   Classes: {len(UNIFIED_CLASSES)}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   data.yaml: {yaml_path}")
    print(f"\nUnified classes:")
    for i, name in enumerate(UNIFIED_CLASSES):
        print(f"   {i}: {name}")


if __name__ == "__main__":
    main()