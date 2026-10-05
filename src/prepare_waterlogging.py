"""Step 2b: convert YOUR waterlogging annotations into YOLO format (class 2).

Waterlogging has no ready-made RDD2022 class, so this data comes from a road-specific
source (a dataset you are licensed to use, or images you annotated yourself).

Two input formats are supported:

  labelme : a folder of images + Labelme .json files (polygons or rectangles).
            Every shape whose label is in --labels becomes a box (polygon -> its bounding box).
              python src/prepare_waterlogging.py --src data_raw/waterlogging --format labelme

  yolo    : a folder with images/ and labels/ already in YOLO format (e.g. a Roboflow export).
            Class IDs listed in --source-ids are rewritten to 2 (waterlogging); others are dropped.
              python src/prepare_waterlogging.py --src data_raw/water_yolo --format yolo --source-ids 0
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from utils import IMAGE_EXTS, STAGING_DIR, find_images, yolo_line

WATER_ID = 2


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def convert_labelme(src, labels):
    out = []
    for js in sorted(src.rglob("*.json")):
        data = json.loads(js.read_text(encoding="utf-8"))
        img = js.parent / data.get("imagePath", "")
        if not img.is_file():   # fall back to same name with an image extension
            img = next((js.with_suffix(e) for e in IMAGE_EXTS if js.with_suffix(e).is_file()), None)
        if img is None:
            print(f"  [skip] no image found for {js.name}")
            continue
        with Image.open(img) as im:
            w, h = im.size
        lines = []
        for shape in data.get("shapes", []):
            if shape.get("label", "").strip().lower() not in labels:
                continue
            xs = [p[0] for p in shape["points"]]
            ys = [p[1] for p in shape["points"]]
            line = yolo_line(WATER_ID, min(xs), min(ys), max(xs), max(ys), w, h)
            if line:
                lines.append(line)
        out.append((img, src_rel(img, src), lines))
    return out


def convert_yolo(src, source_ids):
    out = []
    for img in find_images(src / "images"):
        txt = src / "labels" / (img.stem + ".txt")
        lines = []
        if txt.is_file():
            for raw in txt.read_text(encoding="utf-8").splitlines():
                parts = raw.split()
                if len(parts) == 5 and int(parts[0]) in source_ids:
                    lines.append(" ".join([str(WATER_ID)] + parts[1:]))
        out.append((img, src_rel(img, src), lines))
    return out


def src_rel(img, src):
    return safe_name("_".join(img.relative_to(src).with_suffix("").parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--format", choices=["labelme", "yolo"], required=True)
    ap.add_argument("--labels", nargs="+", default=["waterlogging", "waterlogged", "water"],
                    help="Labelme label names that mean waterlogging (lower-case)")
    ap.add_argument("--source-ids", nargs="+", type=int, default=[0],
                    help="YOLO class IDs in the source data that mean waterlogging")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"Folder not found: {src}")
    items = convert_labelme(src, set(args.labels)) if args.format == "labelme" \
        else convert_yolo(src, set(args.source_ids))

    img_out, lbl_out = STAGING_DIR / "images", STAGING_DIR / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_img, n_box, n_empty = 0, 0, 0
    for img, stem, lines in items:
        if not lines:
            n_empty += 1
            continue            # an image without water boxes teaches nothing here
        name = "water_" + stem
        shutil.copy2(img, img_out / (name + img.suffix.lower()))
        (lbl_out / (name + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        n_img += 1
        n_box += len(lines)
    print(f"Images written: {n_img}  |  waterlogging boxes: {n_box}  |  skipped (no water boxes): {n_empty}")


if __name__ == "__main__":
    main()
