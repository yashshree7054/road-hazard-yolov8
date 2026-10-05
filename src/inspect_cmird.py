"""Inspect the CMIRD download: find the box-label folders, count boxes per class ID,
and save sample crops for each class ID so YOU can see what each ID means.

Save this file as  src/inspect_cmird.py  and run (adjust the path if needed):
    python src/inspect_cmird.py --src C:\\cmird

Outputs:
    * a text report printed on screen and saved to outputs/cmird_inspect/report.txt
    * outputs/cmird_inspect/class_<ID>.jpg : a grid of sample boxes for each class ID
This script only READS the dataset. It changes nothing.
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "cmird_inspect"


def read_label_file(path):
    """Return (kind, rows). kind = 'box' (5 numbers per line), 'polygon' or 'mixed'/'empty'."""
    rows, kinds = [], set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return "unreadable", []
        if len(vals) == 5:
            kinds.add("box")
            rows.append(vals)
        else:
            kinds.add("polygon")
    if not kinds:
        return "empty", []
    return (kinds.pop() if len(kinds) == 1 else "mixed"), rows


def find_image(label_path, images_by_stem):
    """Match a label to its image: same stem, preferring the sibling images/ folder."""
    parts = list(label_path.parts)
    if "labels" in parts:                       # .../labels/train2017/x.txt -> .../images/train2017/x.*
        i = len(parts) - 1 - parts[::-1].index("labels")
        base = Path(*parts[:i], "images", *parts[i + 1:-1])
        for ext in IMAGE_EXTS:
            cand = base / (label_path.stem + ext)
            if cand.is_file():
                return cand
    options = images_by_stem.get(label_path.stem, [])
    return options[0] if options else None


def make_tile(img, row, size=224):
    """Crop a square window around the box, outline the box, resize to size x size."""
    h, w = img.shape[:2]
    _, xc, yc, bw, bh = row
    x1, y1, x2, y2 = (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h
    side = int(min(max(max(x2 - x1, y2 - y1) * 2.0, 96), min(h, w)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(min(max(cx - side / 2, 0), w - side))
    top = int(min(max(cy - side / 2, 0), h - side))
    crop = img[top:top + side, left:left + side].copy()
    cv2.rectangle(crop, (int(x1 - left), int(y1 - top)), (int(x2 - left), int(y2 - top)), (0, 0, 255), 2)
    return cv2.resize(crop, (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder where CMIRD was extracted")
    ap.add_argument("--samples", type=int, default=12, help="sample crops per class ID")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=int, nargs="+", default=None, help="only make sheets for these class IDs")
    ap.add_argument("--tile", type=int, default=224, help="size of each sample tile in pixels")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"Folder not found: {src}")

    all_files = [p for p in src.rglob("*") if p.is_file()]
    images_by_stem = defaultdict(list)
    for p in all_files:
        if p.suffix.lower() in IMAGE_EXTS:
            images_by_stem[p.stem].append(p)
    label_files = [p for p in all_files if p.suffix.lower() == ".txt" and p.name.lower() != "readme.txt"]

    # Group label files by their grandparent-or-parent folder up to 'labels' (one group per dataset part)
    groups = defaultdict(list)
    for p in label_files:
        parts = p.relative_to(src).parts
        key = "/".join(parts[:parts.index("labels")]) if "labels" in parts else "/".join(parts[:-1])
        groups[key].append(p)

    report = [f"Images found: {sum(len(v) for v in images_by_stem.values())} | label .txt files: {len(label_files)}", ""]
    candidates = defaultdict(list)       # (folder, class id) -> [(label_path, row)]
    dims = defaultdict(list)             # (folder, class id) -> [(width, height)] as fractions of the image
    prefixes = defaultdict(Counter)      # folder -> Counter of file-name prefixes
    for key in sorted(groups):
        counts, kinds, with_image = Counter(), Counter(), 0
        for p in groups[key]:
            kind, rows = read_label_file(p)
            kinds[kind] += 1
            with_image += find_image(p, images_by_stem) is not None
            for row in rows:
                counts[int(row[0])] += 1
                candidates_key = (key, int(row[0]))
                if kind == "box":
                    candidates[candidates_key].append((p, row))
                    dims[candidates_key].append((row[3], row[4]))
            prefixes[key][p.stem.split("_")[0] if "_" in p.stem else p.stem[:6]] += 1
        report.append(f"FOLDER: {key}")
        report.append("   file-name prefixes (top 8): " + ", ".join(f"{k}: {v}" for k, v in prefixes[key].most_common(8)))
        report.append(f"   label files: {len(groups[key])}  types: {dict(kinds)}  with matching image: {with_image}")
        if counts:
            report.append("   boxes per class ID: " + ", ".join(f"{k}: {counts[k]}" for k in sorted(counts)))
            for cid in sorted(counts):
                d = dims.get((key, cid))
                if d:
                    ws, hs = sorted(x[0] for x in d), sorted(x[1] for x in d)
                    mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
                    report.append(f"      ID {cid}: median box = {mw:.3f} wide x {mh:.3f} high (fraction of image), width/height = {mw / mh:.2f}")
        report.append("")

    # Sample crops per (folder, class id), only for folders that contain box labels
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    box_keys = sorted(candidates)
    folders_with_boxes = sorted({k[0] for k in box_keys})
    if folders_with_boxes:
        # Use the box folder with the most label files for the class sheets
        main = max(folders_with_boxes, key=lambda f: len(groups[f]))
        report.append(f"Class sheets are made from: {main}")
        for (folder, cid) in box_keys:
            if folder != main or (args.only and cid not in args.only):
                continue
            pool = candidates[(folder, cid)]
            areas = sorted(r[3] * r[4] for _, r in pool)
            median = areas[len(areas) // 2]
            big = [x for x in pool if x[1][3] * x[1][4] >= median] or pool   # skip tiny boxes
            tiles = []
            for lp, row in rng.sample(big, min(args.samples, len(big))):
                img_path = find_image(lp, images_by_stem)
                img = cv2.imread(str(img_path)) if img_path else None
                if img is not None:
                    tiles.append(make_tile(img, row, args.tile))
            if not tiles:
                continue
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
            cv2.putText(grid, f"class ID {cid}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4)
            cv2.putText(grid, f"class ID {cid}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.imwrite(str(OUT_DIR / f"class_{cid}.jpg"), grid)
        report.append(f"Class sample sheets saved in: {OUT_DIR}")
    else:
        report.append("No box-format label files found.")

    text = "\n".join(report)
    print(text)
    (OUT_DIR / "report.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
