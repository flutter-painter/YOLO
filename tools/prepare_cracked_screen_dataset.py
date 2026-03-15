#!/usr/bin/env python3
"""
Prepare the Roboflow "Cracked Screen" export for YOLOv9 training in this repo.

- Copies images from Roboflow train/valid into data/cracked_screen/images/train and images/valid.
- Converts standard YOLO xywh labels to this repo's xyxy .txt format into data/cracked_screen/labels.

Run from project root:
  python tools/prepare_cracked_screen_dataset.py

Source is expected at: "Cracked Screen.v1i.yolov9" (relative to project root).
Output: data/cracked_screen/
"""

import shutil
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tools.convert_yolo_xywh_to_xyxy_txt import convert_file

SOURCE = PROJECT_ROOT / "Cracked Screen.v1i.yolov9"
OUTPUT = PROJECT_ROOT / "data" / "cracked_screen"
SPLITS = ("train", "valid")


def main() -> None:
    if not SOURCE.is_dir():
        print(f"Source not found: {SOURCE}")
        print("Extract the Roboflow dataset so that 'Cracked Screen.v1i.yolov9' exists in the project root.")
        sys.exit(1)

    for split in SPLITS:
        src_images = SOURCE / split / "images"
        src_labels = SOURCE / split / "labels"
        dst_images = OUTPUT / "images" / split
        dst_labels = OUTPUT / "labels" / split

        if not src_images.is_dir():
            print(f"Skipping {split}: no {src_images}")
            continue

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        # Copy images (keep same filenames so labels match)
        for img_path in src_images.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy2(img_path, dst_images / img_path.name)
        print(f"Copied {len(list(dst_images.iterdir()))} images -> {dst_images.relative_to(PROJECT_ROOT)}")

        # Convert labels (xywh -> xyxy) into dst_labels
        if src_labels.is_dir():
            for txt_path in sorted(src_labels.glob("*.txt")):
                out_path = dst_labels / txt_path.name
                convert_file(txt_path, out_path)
            print(f"Converted {len(list(dst_labels.iterdir()))} labels -> {dst_labels.relative_to(PROJECT_ROOT)}")

    print(f"\nDataset ready at: {OUTPUT}")
    print("Train with: python yolo/lazy.py task=train dataset=cracked_screen task.data.batch_size=8")


if __name__ == "__main__":
    main()
