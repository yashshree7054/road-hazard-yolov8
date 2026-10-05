"""Shared settings and helper functions used by every script in this project.

All paths are computed relative to the project root (the folder that contains
`src/`), so the project works on any computer without editing paths.
"""
import csv
import platform
from pathlib import Path

import yaml

# ----------------------------------------------------------------------------
# Project-wide settings
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The order here defines the class IDs written into the label files:
# 0 = pothole, 1 = crack, 2 = waterlogging
CLASS_NAMES = ["pothole", "crack", "waterlogging"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STAGING_DIR = PROJECT_ROOT / "dataset_staging"   # converted, not yet split
DATASET_DIR = PROJECT_ROOT / "dataset"           # final train/val/test split
DATA_YAML = PROJECT_ROOT / "data.yaml"           # generated, do not edit by hand
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPERIMENT_LOG = RUNS_DIR / "experiments.csv"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def find_images(folder):
    """Return a sorted list of image files inside `folder` (searched recursively)."""
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def yolo_line(class_id, xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert a pixel box (xmin, ymin, xmax, ymax) to one YOLO label line.

    YOLO format: `class x_center y_center width height`, all divided by the
    image size so every number is between 0 and 1.
    Returns None if the box is empty after being clipped to the image.
    """
    xmin, xmax = max(0.0, min(xmin, xmax)), min(float(img_w), max(xmin, xmax))
    ymin, ymax = max(0.0, min(ymin, ymax)), min(float(img_h), max(ymin, ymax))
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 1 or bh <= 1:          # smaller than one pixel: ignore
        return None
    xc, yc = xmin + bw / 2, ymin + bh / 2
    return f"{class_id} {xc / img_w:.6f} {yc / img_h:.6f} {bw / img_w:.6f} {bh / img_h:.6f}"


def write_data_yaml():
    """(Re)create data.yaml with the absolute dataset path for THIS computer.

    Ultralytics can resolve a relative `path:` against its own settings folder
    instead of your project, so we write the absolute path at run time.
    """
    content = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    DATA_YAML.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return DATA_YAML


def pick_device():
    """Return 0 (first NVIDIA GPU), 'mps' (Apple chip) or 'cpu'."""
    import torch  # imported here so data-preparation scripts do not need PyTorch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware_description():
    """A short text describing the machine, for the experiment log."""
    import torch

    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} (CUDA)"
    return f"CPU ({platform.processor() or platform.machine()}), {platform.system()}"


def log_experiment(row: dict):
    """Append one experiment (a dict) to runs/experiments.csv."""
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not EXPERIMENT_LOG.exists()
    with open(EXPERIMENT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"Experiment logged to {EXPERIMENT_LOG.relative_to(PROJECT_ROOT)}")
