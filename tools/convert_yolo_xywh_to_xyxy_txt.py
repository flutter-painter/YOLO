#!/usr/bin/env python3
"""
Convert standard YOLO .txt labels (class_id x_center y_center width height, normalized 0-1)
to this repo's expected format: class_id x_min y_min x_max y_max (two corner points).

Usage:
  python tools/convert_yolo_xywh_to_xyxy_txt.py --labels-dir path/to/labels --output-dir data/cracked_screen/labels --splits train val

Input: labels-dir/train/*.txt, labels-dir/val/*.txt (or whatever --splits you pass).
Output: output-dir/train/*.txt, output-dir/val/*.txt with xyxy-style lines.
"""

import argparse
from pathlib import Path


def xywh_to_xyxy_line(parts: list) -> str:
    """Convert one line: class_id x_center y_center width height -> class_id x_min y_min x_max y_max (normalized)."""
    if len(parts) < 5:
        return ""
    cls = parts[0]
    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x_min = max(0.0, xc - w / 2)
    y_min = max(0.0, yc - h / 2)
    x_max = min(1.0, xc + w / 2)
    y_max = min(1.0, yc + h / 2)
    return f"{cls} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f}"


def convert_file(src: Path, dst: Path) -> None:
    """Convert a single .txt file from xywh to xyxy format."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []
    with open(src, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            new_line = xywh_to_xyxy_line(parts)
            if new_line:
                lines_out.append(new_line)
    with open(dst, "w") as f:
        f.write("\n".join(lines_out))
        if lines_out:
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO xywh .txt labels to xyxy format for this repo.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Root dir containing split subdirs (e.g. train/, val/)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for converted labels (split subdirs created)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Subdir names (e.g. train val)")
    args = parser.parse_args()

    for split in args.splits:
        src_dir = args.labels_dir / split
        if not src_dir.is_dir():
            src_dir = args.labels_dir / split / "labels"  # Roboflow-style: train/labels/
        dst_dir = args.output_dir / split
        if not src_dir.is_dir():
            print(f"Skipping (not a dir): {src_dir}")
            continue
        for txt_path in sorted(src_dir.glob("*.txt")):
            out_path = dst_dir / txt_path.name
            convert_file(txt_path, out_path)
            print(f"  {txt_path.relative_to(args.labels_dir)} -> {out_path.relative_to(args.output_dir)}")
    print("Done.")


if __name__ == "__main__":
    main()
