# YOLO26m Multi-Class Aerial Object Detection Pipeline (VisDrone + Seraphim)

A robust, production-grade data pipeline, balanced sampler, training entrypoint, and domain-isolated evaluation suite for aerial object detection across 3 core classes:
- **`0: person`** (Pedestrians and people on foot)
- **`1: vehicle`** (Cars, vans, trucks, buses, motorcycles, bicycles, tricycles)
- **`2: drone`** (Airborne unmanned aerial vehicles)

---

## 📚 Master Guide: How YOLO Data Pipelines Work (So You Can Do It Yourself Next Time)

To build any custom multi-dataset detector in the future without needing AI assistance, you only need to understand **4 fundamental concepts**:

### 1. The Universal YOLO Label Format
Every single object detection image in YOLO requires a corresponding `.txt` file with the exact same filename base (e.g. `frame001.jpg` $\rightarrow$ `frame001.txt`).

Each line inside that text file represents **one bounding box**:
```text
<class_id> <x_center_normalized> <y_center_normalized> <width_normalized> <height_normalized>
```
- **`class_id`**: An integer starting at `0` indexed up to $N-1$.
- **Normalized coordinates**: All coordinates are float numbers between `0.0` and `1.0`, calculated relative to the full image width ($W$) and height ($H$).

#### The Coordinate Conversion Formulas:
If a source dataset gives you bounding box pixel coordinates in top-left format $[x_{min}, y_{min}, \text{width}, \text{height}]$:
$$\text{width}_{norm} = \frac{\text{width}}{W}, \quad \text{height}_{norm} = \frac{\text{height}}{H}$$
$$x_{center} = x_{min} + \frac{\text{width}}{2}, \quad y_{center} = y_{min} + \frac{\text{height}}{2}$$
$$x_{center, norm} = \frac{x_{center}}{W}, \quad y_{center, norm} = \frac{y_{center}}{H}$$

---

### 2. Dataset Anatomy & Class Mapping Decisions

| Source Dataset | Raw Format | Raw Classes | Our Target Mapping | Decision Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **VisDrone2019-DET** | CSV/txt: `[left, top, w, h, score, category, truncation, occlusion]` | `1: pedestrian`, `2: people` | **`0: person`** | Collapsed into a single human class because distinction between walking and standing people is subjective and unnecessary for aerial tracking. |
| **VisDrone2019-DET** | CSV/txt | `3..10: bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor` | **`1: vehicle`** | Collapsed all motorized and non-motorized ground transport into `vehicle`. (Can be toggled to fine-grained via `configs/dataset_config.yaml`). |
| **VisDrone2019-DET** | CSV/txt | `0: ignored regions`, `11: others` | **FILTERED OUT** | Ignored regions and ambiguous "others" are discarded so the model isn't penalized or confused during loss calculation. |
| **Seraphim** | YOLO normalized `.txt` | `0: drone` | **`2: drone`** | Re-indexed class ID from `0` to `2` to prevent label collision with `person`. |

---

### 3. Solving the Imbalance Problem: Ratio Sampling
- **VisDrone**: Dense multi-object ground scenes (~6.4k training images with hundreds of tiny objects per image).
- **Seraphim**: Sparse single-object aerial scenes (~75k training images with 1 drone per image).
- **The Problem**: If you train directly on all 75k Seraphim images, 92% of training batches will contain empty ground scenes and single drones. The gradients for small pedestrians and vehicles will vanish.
- **The Solution**: We implement a **configurable sampling ratio** (e.g. `--drone-ratio 1.0` or `--drone-cap 10000`). This matches the number of drone scenes to ground scenes while logging full class instance distribution before training starts.

---

### 4. Zero-Copy Text Manifests
Rather than copying 80GB+ of image files into a single merged folder, Ultralytics YOLO supports passing a `.txt` file containing absolute paths to each image (e.g., `train_merged.txt`). This allows instant dataset re-sampling and zero disk waste.

---

## 🚀 Step-by-Step Execution Workflow

### Step 0: Installation
```bash
pip install -r requirements.txt
```

---

### Step 1: Download Datasets
Downloads VisDrone2019-DET splits and fetches Seraphim Drone dataset from HuggingFace Hub:
```bash
python data_prep/download_datasets.py
```
*Note: If you have already downloaded either dataset manually, you can place them in `data/raw/visdrone` and `data/raw/seraphim`.*

---

### Step 2: Remap VisDrone Annotations
Converts VisDrone bounding boxes to YOLO normalized coordinates and maps them to `person` (`0`) and `vehicle` (`1`):
```bash
python data_prep/remap_visdrone.py
```
*(Optional: Use `--fine-grained` if you want separate classes for cars, trucks, and 2-wheelers).*

---

### Step 3: Remap Seraphim Annotations
Re-indexes Seraphim drone labels from class `0` $\rightarrow$ class `2` (`drone`):
```bash
python data_prep/remap_seraphim.py
```

---

### Step 4: Build Merged Dataset Manifests & Inspect Statistics

#### Option A: Train on 100% of the Full Dataset (All ~83k Seraphim + All VisDrone Images)
```bash
python data_prep/build_merged_dataset.py --full
```
*This merges 100% of the 75,134 Seraphim training images with all VisDrone training images (totaling ~81,600+ training images).*

#### Option B: Train with Balanced Subsampling (Optional)
```bash
python data_prep/build_merged_dataset.py --drone-ratio 1.0
```

Both options generate the required `.txt` manifests and output dataset statistics to `data/dataset_stats.json`.

---

### Step 5: Visual Sanity Check (Inspect Bounding Boxes)
Render random samples from any dataset with color-coded bounding boxes:
```bash
python data_prep/verify_labels.py --images-dir data/processed/visdrone/images/train --labels-dir data/processed/visdrone/labels/train --output-dir data/inspections/visdrone
python data_prep/verify_labels.py --images-dir data/processed/seraphim/images/train --labels-dir data/processed/seraphim/labels/train --output-dir data/inspections/seraphim
```

---

### Step 6: Train YOLO26m Aerial Detector
Fine-tunes the model starting from COCO pretrained weights with aerial augmentations:
```bash
python train.py --model yolo26m.pt --data configs/merged_aerial.yaml --epochs 120 --batch 16 --imgsz 640 --device 0
```
*(If `yolo26m.pt` is not available in your Ultralytics release, you can use `yolo11m.pt` or `yolov8m.pt` directly via `--model yolo11m.pt`)*.

---

### Step 7: Multi-Domain Isolated Evaluation
Evaluates your trained checkpoint on each domain independently:
```bash
python eval.py --weights runs/detect/yolo26m_aerial_run/weights/best.pt
```

Outputs a comparison report:
```text
============================================================================================
                      PER-DOMAIN PERFORMANCE COMPARISON
============================================================================================
Domain / Val Set          |  Precision |     Recall |     mAP@50 |    mAP@50-95
--------------------------------------------------------------------------------------------
Combined (Merged)         |     0.8410 |     0.7850 |     0.8120 |       0.5630
VisDrone Domain           |     0.8120 |     0.7430 |     0.7840 |       0.5120
Seraphim Domain           |     0.8930 |     0.8520 |     0.8710 |       0.6410
============================================================================================
```
