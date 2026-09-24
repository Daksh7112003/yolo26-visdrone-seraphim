"""
VisDrone2019-DET Label Parser & Remapping Converter.

Converts VisDrone bounding box text files:
  Format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
Into YOLO normalized format:
  Format: <target_class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

Class Mapping:
  - 1 (pedestrian), 2 (people) -> 0 (person)
  - 3 (bicycle), 4 (car), 5 (van), 6 (truck), 7 (tricycle),
    8 (awning-tricycle), 9 (bus), 10 (motor) -> 1 (vehicle)
  - 0 (ignored regions), 11 (others) -> IGNORED / FILTERED OUT
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    VISDRONE_TO_TARGET_MAP,
    VISDRONE_TO_FINE_GRAINED_MAP,
    VISDRONE_RAW_CLASSES,
    TARGET_CLASSES,
    convert_visdrone_bbox_to_yolo
)


def get_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """Get (width, height) of an image without loading full pixel buffer."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        return None


def process_visdrone_split(
    raw_split_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    fine_grained: bool = False,
    symlink_images: bool = False
) -> Dict[str, int]:
    """
    Parses and converts a single VisDrone split (e.g. train or val).
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # In VisDrone, raw directories typically have 'images' and 'annotations'
    images_dir = raw_split_dir / "images"
    annotations_dir = raw_split_dir / "annotations"

    if not images_dir.exists() or not annotations_dir.exists():
        # Check if files are directly inside raw_split_dir
        if (raw_split_dir / "sequences").exists():
            print(f"[!] Warning: {raw_split_dir} appears to be a VisDrone-VID/MOT directory, not VisDrone-DET.")
        print(f"[!] Looking for images/ and annotations/ in {raw_split_dir}")

    # Gather image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions] if images_dir.exists() else []

    if not img_files:
        print(f"[!] No image files found in {images_dir}")
        return {}

    mapping = VISDRONE_TO_FINE_GRAINED_MAP if fine_grained else VISDRONE_TO_TARGET_MAP

    stats = {
        "total_images": len(img_files),
        "total_raw_boxes": 0,
        "valid_mapped_boxes": 0,
        "filtered_ignored_boxes": 0,
        "class_counts": {c: 0 for c in range(len(TARGET_CLASSES))}
    }

    print(f"\n[*] Processing VisDrone split: {raw_split_dir.name} ({len(img_files)} images)...")

    for img_path in tqdm(img_files, desc=f"Converting {raw_split_dir.name}"):
        # Determine destination image path
        dst_img_path = output_images_dir / img_path.name
        if not dst_img_path.exists():
            try:
                os.link(str(img_path.resolve()), str(dst_img_path))
            except Exception:
                try:
                    dst_img_path.symlink_to(img_path.resolve())
                except Exception:
                    import shutil
                    shutil.copy2(img_path, dst_img_path)

        # Corresponding annotation file
        ann_path = annotations_dir / f"{img_path.stem}.txt"
        dst_label_path = output_labels_dir / f"{img_path.stem}.txt"

        if not ann_path.exists():
            # Create an empty label file (valid for images with no target objects)
            dst_label_path.write_text("")
            continue

        img_size = get_image_size(img_path)
        if not img_size:
            continue
        img_w, img_h = img_size

        yolo_lines = []
        with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 6:
                    continue

                stats["total_raw_boxes"] += 1

                try:
                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_width = float(parts[2])
                    bbox_height = float(parts[3])
                    score = int(parts[4])
                    raw_cat = int(parts[5])
                except ValueError:
                    continue

                # In VisDrone, score=0 means ignore in evaluation
                if score == 0:
                    stats["filtered_ignored_boxes"] += 1
                    continue

                # Map class ID
                target_cat = mapping.get(raw_cat, None)
                if target_cat is None:
                    stats["filtered_ignored_boxes"] += 1
                    continue

                # Convert bbox coordinates
                norm_coords = convert_visdrone_bbox_to_yolo(
                    bbox_left=bbox_left,
                    bbox_top=bbox_top,
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    img_width=img_w,
                    img_height=img_h
                )

                if norm_coords is None:
                    stats["filtered_ignored_boxes"] += 1
                    continue

                xc, yc, nw, nh = norm_coords
                yolo_lines.append(f"{target_cat} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                stats["valid_mapped_boxes"] += 1
                if target_cat in stats["class_counts"]:
                    stats["class_counts"][target_cat] += 1

        # Write converted label file
        with open(dst_label_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format with class remapping.")
    parser.add_argument("--raw-visdrone-dir", type=str, default=str(RAW_DIR / "visdrone"),
                        help="Path to raw VisDrone directory containing VisDrone2019-DET-train/val")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR / "visdrone"),
                        help="Path to output processed directory")
    parser.add_argument("--fine-grained", action="store_true",
                        help="Use 5-class fine-grained vehicle taxonomy instead of 3-class")
    parser.add_argument("--symlink", action="store_true",
                        help="Create symlinks for images instead of copying")

    args = parser.parse_args()
    raw_root = Path(args.raw_visdrone_dir)
    out_root = Path(args.output_dir)

    print("=" * 60)
    print("VISDRONE2019-DET LABEL REMAPPER & CONVERTER")
    print(f"Target Taxonomy: {'5-Class Fine-Grained' if args.fine_grained else '3-Class [person, vehicle, drone]'}")
    print("=" * 60)

    splits = ["train", "val", "test-dev"]
    for split in splits:
        split_raw = raw_root / f"VisDrone2019-DET-{split}"
        if not split_raw.exists():
            continue

        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split

        stats = process_visdrone_split(
            raw_split_dir=split_raw,
            output_images_dir=out_img,
            output_labels_dir=out_lbl,
            fine_grained=args.fine_grained,
            symlink_images=args.symlink
        )

        if stats:
            print(f"\n--- Split '{split}' Summary ---")
            print(f"  Total Images Processed   : {stats['total_images']:,}")
            print(f"  Total Raw BBoxes Read    : {stats['total_raw_boxes']:,}")
            print(f"  Valid YOLO Mapped BBoxes : {stats['valid_mapped_boxes']:,}")
            print(f"  Filtered Ignored/Invalid : {stats['filtered_ignored_boxes']:,}")
            print("  Class Breakdown:")
            for c_id, count in stats["class_counts"].items():
                c_name = TARGET_CLASSES[c_id] if c_id < len(TARGET_CLASSES) else f"class_{c_id}"
                print(f"    - [{c_id}] {c_name:<10}: {count:,} instances")


if __name__ == "__main__":
    main()
