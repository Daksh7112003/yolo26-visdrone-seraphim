"""
Multi-Domain Isolated Evaluation Script for Aerial Object Detection.

Evaluates trained YOLO checkpoint separately on:
1. VisDrone Domain Validation Set (dense ground pedestrians & vehicles)
2. Seraphim Domain Validation Set (airborne drones)
3. Merged / Combined Validation Set
4. Real-Footage Skydroid C12 Validation Set (if provided)

Generates comparison tables for mAP50, mAP50-95, Precision, and Recall per domain.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def evaluate_domain(
    model: Any,
    data_yaml_path: Path,
    domain_name: str,
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = ""
) -> Optional[Dict[str, Any]]:
    """Runs Ultralytics validation on a specific domain yaml and extracts metric dict."""
    if not data_yaml_path.exists():
        print(f"[!] Warning: Config file not found for domain '{domain_name}' at {data_yaml_path}")
        return None

    print(f"\n[*] Evaluating Domain: '{domain_name}' using {data_yaml_path.name}...")
    val_kwargs = {
        "data": str(data_yaml_path),
        "imgsz": imgsz,
        "batch": batch,
        "conf": conf,
        "iou": iou,
        "verbose": False,
        "plots": False
    }
    if device:
        val_kwargs["device"] = device

    try:
        metrics = model.val(**val_kwargs)
    except Exception as e:
        print(f"[!] Error during evaluation of '{domain_name}': {e}")
        return None

    # Extract overall metrics
    mp = metrics.box.mp
    mr = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    # Extract class-level metrics
    class_map50 = metrics.box.all_ap[:, 0] if hasattr(metrics.box, 'all_ap') else metrics.box.ap50
    class_names = metrics.names

    class_metrics = {}
    for idx, name in class_names.items():
        if idx < len(metrics.box.ap50):
            c_p = metrics.box.p[idx] if hasattr(metrics.box, 'p') and idx < len(metrics.box.p) else 0.0
            c_r = metrics.box.r[idx] if hasattr(metrics.box, 'r') and idx < len(metrics.box.r) else 0.0
            c_map50 = metrics.box.ap50[idx] if idx < len(metrics.box.ap50) else 0.0
            c_map = metrics.box.ap[idx] if hasattr(metrics.box, 'ap') and idx < len(metrics.box.ap) else 0.0
            class_metrics[name] = {
                "precision": float(c_p),
                "recall": float(c_r),
                "mAP50": float(c_map50),
                "mAP50_95": float(c_map)
            }

    return {
        "domain": domain_name,
        "precision": float(mp),
        "recall": float(mr),
        "mAP50": float(map50),
        "mAP50_95": float(map50_95),
        "class_breakdown": class_metrics
    }


def print_comparison_table(results_list: List[Dict[str, Any]], target_classes: List[str]):
    """Renders formatted multi-domain comparison table."""
    print("\n" + "=" * 92)
    print("                      PER-DOMAIN PERFORMANCE COMPARISON")
    print("=" * 92)

    # Domain overall table
    header = f"{'Domain / Val Set':<25} | {'Precision':>10} | {'Recall':>10} | {'mAP@50':>10} | {'mAP@50-95':>12}"
    print(header)
    print("-" * len(header))

    for res in results_list:
        if not res:
            continue
        print(f"{res['domain']:<25} | {res['precision']:>10.4f} | {res['recall']:>10.4f} | {res['mAP50']:>10.4f} | {res['mAP50_95']:>12.4f}")

    print("-" * len(header))

    # Per-Class Breakdown Across Domains
    print("\n" + "=" * 92)
    print("                       PER-CLASS mAP@50 ACROSS DOMAINS")
    print("=" * 92)
    cls_header = f"{'Domain':<25} | " + " | ".join([f"{c_name + ' mAP50':>18}" for c_name in target_classes])
    print(cls_header)
    print("-" * len(cls_header))

    for res in results_list:
        if not res:
            continue
        row_str = f"{res['domain']:<25} | "
        cls_entries = []
        for c_name in target_classes:
            val = res["class_breakdown"].get(c_name, {}).get("mAP50", 0.0)
            cls_entries.append(f"{val:>18.4f}")
        row_str += " | ".join(cls_entries)
        print(row_str)

    print("=" * 92 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO aerial model with per-domain breakdown.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model weights checkpoint (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for evaluation")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for validation")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="", help="Device: '0', 'cpu'")
    parser.add_argument("--output-json", type=str, default="runs/domain_eval_results.json",
                        help="Output path for evaluation JSON report")

    args = parser.parse_args()

    from ultralytics import YOLO
    project_root = Path(__file__).resolve().parent

    print(f"[*] Loading model checkpoint: {args.weights}")
    model = YOLO(args.weights)

    domains = [
        ("Combined (Merged)", project_root / "configs" / "merged_aerial.yaml"),
        ("VisDrone Domain", project_root / "configs" / "visdrone_val.yaml"),
        ("Seraphim Domain", project_root / "configs" / "seraphim_val.yaml"),
    ]

    results = []
    target_classes = ["person", "vehicle", "drone"]

    for domain_name, config_path in domains:
        res = evaluate_domain(
            model=model,
            data_yaml_path=config_path,
            domain_name=domain_name,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device
        )
        if res:
            results.append(res)

    if results:
        print_comparison_table(results, target_classes)

        # Save JSON output
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[✓] Domain evaluation metrics saved to: {out_p}")


if __name__ == "__main__":
    main()
