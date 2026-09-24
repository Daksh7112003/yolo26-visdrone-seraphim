"""
Seraphim Drone Dataset Class-ID Remapper.

Seraphim annotations are already in standard YOLO format:
  Format: <class_id> <x_center> <y_center> <width> <height>
Where original class_id is 0 (drone).

This script:
1. Re-indexes class_id from 0 -> 2 (or target drone index).
2. Verifies normalized bounding box validity (0.0 <= val <= 1.0).
3. Preserves original train/test/val splits into the processed directory.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import RAW_DIR, PROCESSED_DIR, SERAPHIM_TARGET_CLASS_ID, TARGET_CLASSES


import zipfile

def extract_seraphim_zips_if_needed(base_dir: Path):
    """Auto-extracts any batch_*.zip files found inside Seraphim directories."""
    zip_files = list(base_dir.glob("**/*.zip"))
    if not zip_files:
        return

    print(f"[*] Found {len(zip_files)} ZIP archives in Seraphim raw directory. Extracting...")
    for zf in tqdm(zip_files, desc="Extracting Seraphim archives"):
        # Check if already extracted
        target_dir = zf.parent
        try:
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                # Check if sample file already extracted
                namelist = zip_ref.namelist()
                if namelist and (target_dir / namelist[0]).exists():
                    continue
                zip_ref.extractall(target_dir)
        except Exception as e:
            print(f"[!] Error extracting {zf.name}: {e}")


def find_split_directories(base_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Auto-detects image and label directories in various Seraphim dataset layouts.
    Handles flat structures, train/val/test splits, or images/labels hierarchies.
    """
    extract_seraphim_zips_if_needed(base_dir)

    splits = {}

    candidate_splits = ["train", "test", "val", "valid"]

    for split in candidate_splits:
        img_candidates = [
            base_dir / split / "images",
            base_dir / "images" / split,
            base_dir / split
        ]
        lbl_candidates = [
            base_dir / split / "labels",
            base_dir / "labels" / split,
            base_dir / split
        ]

        found_img = None
        for p in img_candidates:
            if p.is_dir():
                found_img = p
                break

        found_lbl = None
        for p in lbl_candidates:
            if p.is_dir():
                found_lbl = p
                break

        if found_img and found_lbl:
            split_key = "val" if split in ["test", "valid"] else split
            splits[split_key] = {"images": found_img, "labels": found_lbl}

    return splits


def remap_seraphim_split(
    img_dir: Path,
    lbl_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    target_drone_id: int = SERAPHIM_TARGET_CLASS_ID,
    symlink_images: bool = False
) -> Dict[str, int]:
    """Remaps class IDs for all label files in a Seraphim split."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in img_extensions]

    stats = {
        "total_images": len(img_files),
        "total_instances": 0,
        "remapped_instances": 0,
        "empty_or_background_images": 0
    }

    for img_path in tqdm(img_files, desc=f"Remapping {img_dir.name}"):
        # Link or copy image
        dst_img_path = out_img_dir / img_path.name
        if not dst_img_path.exists():
            try:
                os.link(str(img_path.resolve()), str(dst_img_path))
            except Exception:
                try:
                    dst_img_path.symlink_to(img_path.resolve())
                except Exception:
                    shutil.copy2(img_path, dst_img_path)

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        dst_lbl_path = out_lbl_dir / f"{img_path.stem}.txt"

        if not lbl_path.exists():
            dst_lbl_path.write_text("")
            stats["empty_or_background_images"] += 1
            continue

        remapped_lines = []
        with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue

                stats["total_instances"] += 1
                try:
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue

                # Ensure values within bounds
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                remapped_lines.append(f"{target_drone_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                stats["remapped_instances"] += 1

        if not remapped_lines:
            stats["empty_or_background_images"] += 1

        with open(dst_lbl_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Remap Seraphim Drone class labels to target class ID.")
    parser.add_argument("--raw-seraphim-dir", type=str, default=str(RAW_DIR / "seraphim"),
                        help="Path to raw Seraphim directory")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR / "seraphim"),
                        help="Path to output processed directory")
    parser.add_argument("--drone-class-id", type=int, default=SERAPHIM_TARGET_CLASS_ID,
                        help=f"Target class ID for drone (default: {SERAPHIM_TARGET_CLASS_ID})")
    parser.add_argument("--symlink", action="store_true", help="Symlink images instead of copying")

    args = parser.parse_args()
    raw_dir = Path(args.raw_seraphim_dir)
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("SERAPHIM DRONE DATASET CLASS REMAPPER")
    print(f"Target Drone Class ID: {args.drone_class_id} ({TARGET_CLASSES[args.drone_class_id]})")
    print("=" * 60)

    splits = find_split_directories(raw_dir)
    if not splits:
        print(f"[!] No valid split directories detected in {raw_dir}")
        print("    Ensure Seraphim images/labels are downloaded into raw_dir.")
        return

    for split_name, paths in splits.items():
        print(f"\n[*] Processing split: '{split_name}'...")
        out_img = out_dir / "images" / split_name
        out_lbl = out_dir / "labels" / split_name

        stats = remap_seraphim_split(
            img_dir=paths["images"],
            lbl_dir=paths["labels"],
            out_img_dir=out_img,
            out_lbl_dir=out_lbl,
            target_drone_id=args.drone_class_id,
            symlink_images=args.symlink
        )

        print(f"\n--- Seraphim '{split_name}' Summary ---")
        print(f"  Total Images Processed : {stats['total_images']:,}")
        print(f"  Remapped Drone Boxes   : {stats['remapped_instances']:,}")
        print(f"  Background Images      : {stats['empty_or_background_images']:,}")


if __name__ == "__main__":
    main()
