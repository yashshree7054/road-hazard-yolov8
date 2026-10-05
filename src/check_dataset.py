"""Step 4: quality check of the final dataset/ folder.

    python src/check_dataset.py

Reports: images per split, boxes per class per split (class imbalance),
corrupted images, images without a label file, label files without an image,
and malformed label lines (wrong class ID, values outside 0-1).
"""
from collections import Counter
from pathlib import Path

from PIL import Image

from utils import CLASS_NAMES, DATASET_DIR, find_images


def main():
    if not DATASET_DIR.is_dir():
        raise SystemExit("dataset/ not found. Run split_dataset.py first.")

    problems = 0
    for split in ("train", "val", "test"):
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        if not img_dir.is_dir():
            print(f"[{split}] missing")
            continue
        images = find_images(img_dir)
        per_class = Counter()
        no_label = corrupted = bad_lines = 0

        for img in images:
            try:
                with Image.open(img) as im:
                    im.verify()                      # raises if the file is damaged
            except Exception as err:                 # report the real reason
                print(f"  CORRUPTED: {img.name} ({err})")
                corrupted += 1
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.is_file():
                no_label += 1
                continue
            for n, raw in enumerate(lbl.read_text(encoding="utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                parts = raw.split()
                try:
                    cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
                    ok = len(vals) == 4 and 0 <= cls < len(CLASS_NAMES) and all(0 <= v <= 1 for v in vals)
                except ValueError:
                    ok = False
                if ok:
                    per_class[cls] += 1
                else:
                    print(f"  BAD LABEL: {lbl.name} line {n}: {raw!r}")
                    bad_lines += 1

        orphan = [p for p in lbl_dir.glob("*.txt") if not any((img_dir / (p.stem + e)).exists()
                  for e in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))] if lbl_dir.is_dir() else []
        print(f"\n[{split}] images: {len(images)}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"   {name:<13} boxes: {per_class[i]}")
        print(f"   no label file: {no_label} | corrupted: {corrupted} | bad lines: {bad_lines} | orphan labels: {len(orphan)}")
        problems += corrupted + bad_lines + len(orphan)

    print("\nResult:", "no hard problems found" if problems == 0 else f"{problems} problem(s) to fix above")
    print("Tip: if one class has far fewer boxes than the others, expect weaker results for it.")


if __name__ == "__main__":
    main()
