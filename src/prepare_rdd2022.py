"""Step 2a: convert RDD2022 (Pascal VOC XML labels) into YOLO format with OUR 3 classes.

Label mapping (RDD2022 code -> our class):
    D40 (pothole)             -> 0 pothole
    D00 (longitudinal crack)  -> 1 crack
    D10 (transverse crack)    -> 1 crack
    D20 (alligator crack)     -> 1 crack
Any other code found in the XML files is NOT mapped; it is listed in the report
and ignored, so nothing is merged blindly.

Typical use (inspect first, then convert):
    python src/prepare_rdd2022.py --src data_raw/RDD2022/India --dry-run
    python src/prepare_rdd2022.py --src data_raw/RDD2022/India

Expected input layout (please verify against your download):
    <src>/train/images/*.jpg
    <src>/train/annotations/xmls/*.xml
"""
import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

from utils import CLASS_NAMES, IMAGE_EXTS, STAGING_DIR, yolo_line

LABEL_MAP = {"D40": 0, "D00": 1, "D10": 1, "D20": 1}


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def find_image(xml_path, filename):
    """Look for the image that belongs to an XML file."""
    candidates = [
        xml_path.parent.parent.parent / "images" / filename,   # .../annotations/xmls/x.xml
        xml_path.parent / filename,
    ]
    stem = xml_path.stem
    for folder in (xml_path.parent.parent.parent / "images", xml_path.parent):
        candidates += [folder / (stem + ext) for ext in IMAGE_EXTS]
    return next((c for c in candidates if c.is_file()), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="RDD2022 folder (or one country folder)")
    ap.add_argument("--dry-run", action="store_true", help="only count classes, write nothing")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also keep images with none of our classes (as background images)")
    args = ap.parse_args()

    src = Path(args.src)
    xmls = sorted(src.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"No .xml files found under {src}. Check the --src path.")

    seen, kept = Counter(), Counter()
    n_images = n_skipped = n_missing_img = 0

    img_out = STAGING_DIR / "images"
    lbl_out = STAGING_DIR / "labels"
    if not args.dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    for xml_path in xmls:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename") or (xml_path.stem + ".jpg")
        objects = root.findall("object")
        for obj in objects:
            seen[(obj.findtext("name") or "").strip()] += 1

        if args.dry_run:
            continue

        img_path = find_image(xml_path, filename)
        if img_path is None:
            n_missing_img += 1
            continue

        # Use the real image size (XML sizes are sometimes missing or wrong)
        with Image.open(img_path) as im:
            w, h = im.size

        lines = []
        for obj in objects:
            code = (obj.findtext("name") or "").strip()
            if code not in LABEL_MAP:
                continue
            bb = obj.find("bndbox")
            box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            line = yolo_line(LABEL_MAP[code], *box, w, h)
            if line:
                lines.append(line)
                kept[CLASS_NAMES[LABEL_MAP[code]]] += 1

        if not lines and not args.keep_empty:
            n_skipped += 1
            continue

        # Unique file name: path relative to --src, so countries never collide
        rel = img_path.relative_to(src).with_suffix("")
        new_stem = "rdd_" + safe_name("_".join(rel.parts))
        shutil.copy2(img_path, img_out / (new_stem + img_path.suffix.lower()))
        (lbl_out / (new_stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        n_images += 1

    print("\nClass codes found in the XML files (before mapping):")
    for code, n in seen.most_common():
        target = CLASS_NAMES[LABEL_MAP[code]] if code in LABEL_MAP else "IGNORED (not mapped)"
        print(f"  {code:<6} {n:>7} boxes  ->  {target}")

    if args.dry_run:
        print("\nDry run only: nothing was written.")
        return
    print(f"\nImages written to staging : {n_images}")
    print(f"Images skipped (no target class): {n_skipped}")
    print(f"XML files with no matching image : {n_missing_img}")
    print("Boxes written per class   :", dict(kept))


if __name__ == "__main__":
    main()
