"""
Merged Dataset Builder, Balanced Sampler, and Dataset Statistics Generator.

Creates Ultralytics manifest files (.txt) containing image paths from both
VisDrone and Seraphim without physically duplicating media files.
Enforces configurable dataset balance to prevent the massive Seraphim dataset
from swamping VisDrone's dense small-object scenes.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    DATA_DIR,
    PROCESSED_DIR,
    CONFIGS_DIR,
    TARGET_CLASSES
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images_and_labels(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Pairs existing images with their corresponding label text files."""
    if not images_dir.exists():
        return []

    pairs = []
    for img_path in images_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            pairs.append((img_path.resolve(), lbl_path.resolve() if lbl_path.exists() else None))
    return pairs


def count_instances_in_labels(label_paths: List[Optional[Path]], num_classes: int = len(TARGET_CLASSES)) -> Tuple[Dict[int, int], int]:
    """Scans label files and returns per-class instance counts and background image counts."""
    counts = Counter()
    bg_images = 0

    for lp in label_paths:
        if lp is None or not lp.exists():
            bg_images += 1
            continue

        with open(lp, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                bg_images += 1
                continue

            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        c_id = int(parts[0])
                        counts[c_id] += 1
                    except ValueError:
                        pass

    class_dict = {i: counts.get(i, 0) for i in range(num_classes)}
    return class_dict, bg_images


def build_manifests(
    processed_root: Path,
    output_manifest_dir: Path,
    drone_sampling_ratio: float = 1.0,
    drone_max_cap: Optional[int] = None,
    use_full_dataset: bool = False,
    seed: int = 42
):
    """
    Builds balanced train/val text manifests and logs complete dataset statistics.
    """
    random.seed(seed)
    output_manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1. VisDrone sets
    visdrone_train = collect_images_and_labels(
        processed_root / "visdrone" / "images" / "train",
        processed_root / "visdrone" / "labels" / "train"
    )
    visdrone_val = collect_images_and_labels(
        processed_root / "visdrone" / "images" / "val",
        processed_root / "visdrone" / "labels" / "val"
    )

    # 2. Seraphim sets
    seraphim_train = collect_images_and_labels(
        processed_root / "seraphim" / "images" / "train",
        processed_root / "seraphim" / "labels" / "train"
    )
    seraphim_val = collect_images_and_labels(
        processed_root / "seraphim" / "images" / "val",
        processed_root / "seraphim" / "labels" / "val"
    )

    print("\n" + "=" * 70)
    print("DATASET BALANCING & MANIFEST GENERATION")
    print("=" * 70)
    print(f"Raw Available Counts:")
    print(f"  - VisDrone Train Images : {len(visdrone_train):,}")
    print(f"  - VisDrone Val Images   : {len(visdrone_val):,}")
    print(f"  - Seraphim Train Images : {len(seraphim_train):,}")
    print(f"  - Seraphim Val Images   : {len(seraphim_val):,}")

    # 3. Apply Drone Sampling / Capping (or Full Dataset Mode)
    selected_seraphim_train = list(seraphim_train)
    if seraphim_train and not use_full_dataset:
        target_count = len(seraphim_train)
        if drone_sampling_ratio > 0 and len(visdrone_train) > 0:
            target_count = int(len(visdrone_train) * drone_sampling_ratio)
        if drone_max_cap is not None:
            target_count = min(target_count, drone_max_cap)

        target_count = min(target_count, len(seraphim_train))
        print(f"\n[*] Balancing Strategy Applied:")
        print(f"    - Ratio vs VisDrone : {drone_sampling_ratio:.2f}x")
        print(f"    - Max Cap Limit     : {drone_max_cap}")
        print(f"    - Sampled Seraphim  : {target_count:,} / {len(seraphim_train):,} images")

        # Shuffle deterministically and sample
        random.shuffle(selected_seraphim_train)
        selected_seraphim_train = selected_seraphim_train[:target_count]
    else:
        print(f"\n[*] FULL DATASET MODE: Using 100% of available Seraphim images ({len(seraphim_train):,}) and VisDrone images ({len(visdrone_train):,})")

    # 4. Merge Training Sets
    merged_train_pairs = visdrone_train + selected_seraphim_train
    random.shuffle(merged_train_pairs)

    merged_val_pairs = visdrone_val + seraphim_val
    random.shuffle(merged_val_pairs)

    # 5. Write Manifest Files (.txt)
    manifest_paths = {
        "train_visdrone": output_manifest_dir / "train_visdrone.txt",
        "train_merged": output_manifest_dir / "train_merged.txt",
        "val_merged": output_manifest_dir / "val_merged.txt",
        "val_visdrone": output_manifest_dir / "val_visdrone.txt",
        "val_seraphim": output_manifest_dir / "val_seraphim.txt"
    }

    def write_manifest(file_path: Path, pairs: List[Tuple[Path, Path]]):
        with open(file_path, "w", encoding="utf-8") as f:
            for img_p, _ in pairs:
                # Use forward slashes for cross-platform YOLO compatibility
                f.write(f"{img_p.as_posix()}\n")

    write_manifest(manifest_paths["train_visdrone"], visdrone_train)
    write_manifest(manifest_paths["train_merged"], merged_train_pairs)
    write_manifest(manifest_paths["val_merged"], merged_val_pairs)
    write_manifest(manifest_paths["val_visdrone"], visdrone_val)
    write_manifest(manifest_paths["val_seraphim"], seraphim_val)

    # 6. Compute Comprehensive Instance Statistics
    print("\n" + "=" * 70)
    print("DATASET COMPOSITION & INSTANCE STATISTICS")
    print("=" * 70)

    stats_summary = {}

    datasets_to_analyze = {
        "VisDrone Train": [lp for _, lp in visdrone_train],
        "VisDrone Val": [lp for _, lp in visdrone_val],
        "Seraphim Train (Sampled)": [lp for _, lp in selected_seraphim_train],
        "Seraphim Val": [lp for _, lp in seraphim_val],
        "Merged Train (Final)": [lp for _, lp in merged_train_pairs],
        "Merged Val (Final)": [lp for _, lp in merged_val_pairs],
    }

    # Format table header
    col1 = 28
    col_cls = 14
    header = f"{'Dataset Split':<{col1}}"
    for c_id, c_name in enumerate(TARGET_CLASSES):
        header += f"{f'{c_name} (ID {c_id})':>{col_cls}}"
    header += f"{'Total BBoxes':>{col_cls}}{'BG Images':>{col_cls}}"
    print(header)
    print("-" * len(header))

    for split_title, label_list in datasets_to_analyze.items():
        class_counts, bg_count = count_instances_in_labels(label_list)
        total_boxes = sum(class_counts.values())

        row = f"{split_title:<{col1}}"
        for c_id in range(len(TARGET_CLASSES)):
            row += f"{class_counts[c_id]:>{col_cls},}"
        row += f"{total_boxes:>{col_cls},}{bg_count:>{col_cls},}"
        print(row)

        stats_summary[split_title] = {
            "image_count": len(label_list),
            "background_images": bg_count,
            "total_boxes": total_boxes,
            "class_distribution": {TARGET_CLASSES[i]: class_counts[i] for i in range(len(TARGET_CLASSES))}
        }

    print("-" * len(header))

    # Save stats JSON
    stats_file = DATA_DIR / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"\n[✓] Detailed statistics exported to: {stats_file}")
    print(f"[✓] Manifest files written to: {output_manifest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create balanced dataset manifests for VisDrone + Seraphim.")
    parser.add_argument("--processed-dir", type=str, default=str(PROCESSED_DIR),
                        help="Path to processed dataset root containing visdrone/ and seraphim/")
    parser.add_argument("--output-manifest-dir", type=str, default=str(DATA_DIR / "manifests"),
                        help="Path to save generated manifest .txt files")
    parser.add_argument("--drone-ratio", type=float, default=1.0,
                        help="Sampling ratio of Seraphim images relative to VisDrone train count (default: 1.0)")
    parser.add_argument("--drone-cap", type=int, default=None,
                        help="Optional maximum image limit for Seraphim drone images")
    parser.add_argument("--full", "--use-all", dest="full", action="store_true",
                        help="Use 100% of all Seraphim and VisDrone images (no subsampling/capping)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    build_manifests(
        processed_root=Path(args.processed_dir),
        output_manifest_dir=Path(args.output_manifest_dir),
        drone_sampling_ratio=args.drone_ratio,
        drone_max_cap=args.drone_cap,
        use_full_dataset=args.full,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
