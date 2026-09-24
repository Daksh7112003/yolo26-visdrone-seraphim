"""
YOLO Multi-Class Aerial Object Detector Training Entrypoint.

Trains YOLO26m (or YOLO11m/YOLOv8m) on merged VisDrone + Seraphim aerial dataset.
Features:
- Validates dataset manifests and displays class distribution statistics before launch.
- Fine-tunes from COCO pretrained checkpoint.
- Configures aerial multi-scale, mosaic, and rotation augmentations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def display_dataset_preflight_stats(stats_path: Path):
    """Prints dataset statistics before starting training."""
    if not stats_path.exists():
        print("[!] Note: data/dataset_stats.json not found. Run data_prep/build_merged_dataset.py for detailed balance logs.")
        return

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        print("\n" + "=" * 70)
        print("PRE-TRAINING DATASET SANITY CHECK & CLASS BALANCE")
        print("=" * 70)

        for split_name, details in stats.items():
            img_cnt = details.get("image_count", 0)
            box_cnt = details.get("total_boxes", 0)
            bg_cnt = details.get("background_images", 0)
            classes = details.get("class_distribution", {})

            cls_str = " | ".join([f"{k}: {v:,}" for k, v in classes.items()])
            print(f"• {split_name:<26} : {img_cnt:,} imgs ({bg_cnt:,} bg) | {box_cnt:,} total bboxes")
            print(f"  └─ Distribution: {cls_str}")

        print("=" * 70 + "\n")
    except Exception as e:
        print(f"[!] Warning reading dataset stats: {e}")


def train(
    model_name: str = "yolo26m.pt",
    data_yaml: str = "configs/merged_aerial.yaml",
    epochs: int = 120,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    workers: int = 8,
    cache: Optional[str] = None,
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    lrf: float = 0.01,
    mosaic: float = 1.0,
    mixup: float = 0.15,
    scale: float = 0.5,
    degrees: float = 10.0,
    flipud: float = 0.2,
    fliplr: float = 0.5,
    project: str = "runs/detect",
    name: str = "yolo26m_aerial_seraphim_visdrone",
    exist_ok: bool = False,
    pretrained: bool = True
):
    """Launches Ultralytics YOLO training."""
    from ultralytics import YOLO

    project_root = Path(__file__).resolve().parent
    stats_file = project_root / "data" / "dataset_stats.json"
    data_cfg_path = Path(data_yaml)
    if not data_cfg_path.is_absolute():
        data_cfg_path = project_root / data_yaml

    if not data_cfg_path.exists():
        print(f"[!] Error: Dataset configuration file not found at {data_cfg_path}")
        print("    Please ensure data preparation is complete and yaml exists.")
        sys.exit(1)

    # Display dataset statistics
    display_dataset_preflight_stats(stats_file)

    print(f"[*] Initializing model: '{model_name}' (Pretrained COCO weights: {pretrained})...")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"[!] Checkpoint '{model_name}' could not be resolved directly ({e}).")
        fallback_model = "yolo11m.pt"
        print(f"[*] Falling back to Ultralytics SOTA medium checkpoint: '{fallback_model}'...")
        model = YOLO(fallback_model)

    print(f"[*] Starting aerial detector training on {data_cfg_path.name} for {epochs} epochs...")
    print(f"    - Image Size  : {imgsz}x{imgsz}")
    print(f"    - Batch Size  : {batch}")
    print(f"    - Optimizer   : {optimizer} (lr0={lr0}, lrf={lrf})")
    print(f"    - Device      : {device if device else 'Auto-detect GPU/CPU'}")
    print(f"    - Augmentations: Mosaic={mosaic}, Mixup={mixup}, Scale={scale}, FlipUD={flipud}")

    # High-performance GPU settings to saturate Compute & Tensor Cores
    train_args = {
        "data": str(data_cfg_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "mosaic": mosaic,
        "mixup": mixup,
        "scale": scale,
        "degrees": degrees,
        "flipud": flipud,
        "fliplr": fliplr,
        "project": project,
        "name": name,
        "exist_ok": exist_ok,
        "pretrained": pretrained,
        "verbose": True,
        "val": True,
        "save": True,
        "save_period": 10,
        "patience": 30,
        "amp": True,               # Automatic Mixed Precision for Tensor Core acceleration
        "cache": cache if cache else False,  # Keeps images in RAM to prevent disk I/O GPU stalls
        "close_mosaic": 15,        # Disables mosaic last 15 epochs for stable convergence
    }

    if device:
        train_args["device"] = device

    results = model.train(**train_args)
    print("\n[✓] Training completed successfully!")
    print(f"[✓] Best model checkpoint saved at: {project}/{name}/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26m on Merged VisDrone + Seraphim Dataset.")
    parser.add_argument("--model", type=str, default="yolo26m.pt",
                        help="Base model checkpoint (e.g. yolo26m.pt, yolo11m.pt, yolov8m.pt)")
    parser.add_argument("--data", type=str, default="configs/merged_aerial.yaml",
                        help="Path to dataset configuration YAML")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs (default: 120)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto-batch)")
    parser.add_argument("--device", type=str, default="", help="Device to run on (e.g. '0', '0,1', 'cpu')")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes (default: 8)")
    parser.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "none"],
                        help="Dataset caching mode: 'ram' (fastest, max GPU usage), 'disk', or 'none'")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer: SGD, Adam, AdamW")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.15, help="Mixup augmentation probability")
    parser.add_argument("--scale", type=float, default=0.5, help="Multi-scale jitter (+/- gain)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Save directory root")
    parser.add_argument("--name", type=str, default="yolo26m_aerial_run", help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing experiment directory")

    args = parser.parse_args()

    cache_val = None if args.cache == "none" else (True if args.cache == "ram" else "disk")

    train(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=cache_val,
        optimizer=args.optimizer,
        lr0=args.lr0,
        mosaic=args.mosaic,
        mixup=args.mixup,
        scale=args.scale,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok
    )


if __name__ == "__main__":
    main()
