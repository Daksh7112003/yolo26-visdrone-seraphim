"""
Visual Verification Tool for Converted & Remapped YOLO Labels.

Draws normalized bounding boxes and class labels onto images to verify
coordinate conversions and class mapping correctness.
"""

import argparse
import random
import sys
from pathlib import Path
import cv2

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import DATA_DIR, TARGET_CLASSES, CLASS_COLORS


def draw_yolo_annotations(image_path: Path, label_path: Path, output_path: Path):
    """Reads an image and its YOLO label file, rendering labeled bounding boxes."""
    if not image_path.exists():
        return

    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w, _ = img.shape

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    c_id = int(parts[0])
                    xc_norm = float(parts[1])
                    yc_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                except ValueError:
                    continue

                # Convert normalized YOLO coordinates to absolute pixel box
                bw = w_norm * w
                bh = h_norm * h
                bx1 = int((xc_norm * w) - (bw / 2.0))
                by1 = int((yc_norm * h) - (bh / 2.0))
                bx2 = int(bx1 + bw)
                by2 = int(by1 + bh)

                color = CLASS_COLORS.get(c_id, (255, 255, 255))
                c_name = TARGET_CLASSES[c_id] if c_id < len(TARGET_CLASSES) else f"class_{c_id}"

                # Draw bounding box
                cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)

                # Draw class label background and text
                label_text = f"{c_name} ({c_id})"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (bx1, max(0, by1 - th - baseline - 4)), (bx1 + tw, max(th + baseline + 4, by1)), color, -1)
                cv2.putText(img, label_text, (bx1, max(th + 2, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def verify_dataset(images_dir: Path, labels_dir: Path, output_dir: Path, num_samples: int = 10):
    """Selects random samples from dataset and generates annotated inspection images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in img_extensions]

    if not images:
        print(f"[!] No images found in {images_dir}")
        return

    samples = random.sample(images, min(num_samples, len(images)))
    print(f"[*] Generating {len(samples)} visual verification samples -> {output_dir}")

    for img_p in samples:
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        out_p = output_dir / f"annotated_{img_p.name}"
        draw_yolo_annotations(img_p, lbl_p, out_p)

    print(f"[✓] Verification images saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visually verify YOLO label conversions on images.")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--labels-dir", type=str, required=True, help="Directory containing YOLO txt labels")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DIR / "inspections"),
                        help="Directory to save rendered sample images")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of random samples to render")

    args = parser.parse_args()
    verify_dataset(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
