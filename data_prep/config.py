"""
Central Configuration & Class Mapping Registry for Aerial Object Detection Pipeline.
Handles class ID mapping between VisDrone2019-DET, Seraphim, and Target YOLO classes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Default Target 3-Class Taxonomy
TARGET_CLASSES: List[str] = [
    "person",   # Class Index 0
    "vehicle",  # Class Index 1
    "drone"     # Class Index 2
]

# Color map for bounding box visualization (BGR format for OpenCV)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),    # Green  -> Person
    1: (255, 128, 0),  # Blue/Cyan -> Vehicle
    2: (0, 0, 255)     # Red    -> Drone
}

# ==============================================================================
# VISDRONE 2019-DET SOURCE CLASS DEFINITIONS
# VisDrone raw class IDs:
#   0: ignored regions
#   1: pedestrian
#   2: people
#   3: bicycle
#   4: car
#   5: van
#   6: truck
#   7: tricycle
#   8: awning-tricycle
#   9: bus
#   10: motor
#   11: others
# ==============================================================================

VISDRONE_RAW_CLASSES = {
    0: "ignored_regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others"
}

# Standard 3-Class mapping (VisDrone -> Target Class Index)
# None indicates the category should be filtered out / ignored.
VISDRONE_TO_TARGET_MAP: Dict[int, Optional[int]] = {
    0: None,  # ignored regions -> skip
    1: 0,     # pedestrian -> person
    2: 0,     # people -> person
    3: 1,     # bicycle -> vehicle
    4: 1,     # car -> vehicle
    5: 1,     # van -> vehicle
    6: 1,     # truck -> vehicle
    7: 1,     # tricycle -> vehicle
    8: 1,     # awning-tricycle -> vehicle
    9: 1,     # bus -> vehicle
    10: 1,    # motor -> vehicle
    11: None  # others -> skip
}

# Optional Fine-Grained Vehicle Taxonomy (If finer granularity is desired in the future)
FINE_GRAINED_TARGET_CLASSES = [
    "person",       # 0
    "car_van",      # 1
    "truck_bus",    # 2
    "two_wheeler",  # 3 (bicycle, motor, tricycle, awning-tricycle)
    "drone"         # 4
]

VISDRONE_TO_FINE_GRAINED_MAP: Dict[int, Optional[int]] = {
    0: None,  # ignored
    1: 0,     # pedestrian -> person
    2: 0,     # people -> person
    3: 3,     # bicycle -> two_wheeler
    4: 1,     # car -> car_van
    5: 1,     # van -> car_van
    6: 2,     # truck -> truck_bus
    7: 3,     # tricycle -> two_wheeler
    8: 3,     # awning-tricycle -> two_wheeler
    9: 2,     # bus -> truck_bus
    10: 3,    # motor -> two_wheeler
    11: None  # others -> skip
}

# ==============================================================================
# SERAPHIM DRONE DATASET MAPPING
# In Seraphim, drones are usually class 0. We map drone -> Index 2 (or Index 4 in fine-grained)
# ==============================================================================
SERAPHIM_DRONE_RAW_CLASS_ID = 0
SERAPHIM_TARGET_CLASS_ID = 2


def convert_visdrone_bbox_to_yolo(
    bbox_left: float,
    bbox_top: float,
    bbox_width: float,
    bbox_height: float,
    img_width: int,
    img_height: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    Converts VisDrone absolute [bbox_left, bbox_top, bbox_width, bbox_height]
    to YOLO normalized [x_center, y_center, width, height].

    Args:
        bbox_left: X-coordinate of top-left corner
        bbox_top: Y-coordinate of top-left corner
        bbox_width: Width of bounding box
        bbox_height: Height of bounding box
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x_center_norm, y_center_norm, width_norm, height_norm) or None if invalid.
    """
    if img_width <= 0 or img_height <= 0:
        return None

    # Filter out non-positive dimensions
    if bbox_width <= 0 or bbox_height <= 0:
        return None

    # Clip coordinates to image boundary
    x1 = max(0.0, min(float(bbox_left), float(img_width)))
    y1 = max(0.0, min(float(bbox_top), float(img_height)))
    x2 = max(0.0, min(float(bbox_left + bbox_width), float(img_width)))
    y2 = max(0.0, min(float(bbox_top + bbox_height), float(img_height)))

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None

    # Calculate center point
    xc = x1 + (w / 2.0)
    yc = y1 + (h / 2.0)

    # Normalize to [0.0, 1.0]
    xc_norm = xc / img_width
    yc_norm = yc / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Final sanity bounds check
    xc_norm = max(0.0, min(1.0, xc_norm))
    yc_norm = max(0.0, min(1.0, yc_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return xc_norm, yc_norm, w_norm, h_norm
