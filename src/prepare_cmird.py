"""Convert the CMIRD detection labels into OUR 3-class YOLO format.

CMIRD class IDs (alphabetical order of the class names; checked on sample images):
    0 auto-rickshaw  1 bike  2 bus  3 car  4 crack  5 manhole  6 person  7 pothole  8 speed bump  9 waterlogging
Our classes:  0 pothole  1 crack  2 waterlogging
Mapping:      CMIRD 7 -> 0,  CMIRD 4 -> 1,  CMIRD 9 -> 2.  Everything else is dropped.
Only the 'detection-object' labels are used (the segmentation folders are ignored).

First look without writing anything:
    python src/prepare_cmird.py --src C:\\cmird --dry-run
Then convert:
    python src/prepare_cmird.py --src C:\\cmird
Output goes to dataset_staging/ (same place as the other prepare scripts), plus a
manifest file dataset_staging/manifest_cmird.csv recording each image's original split.
"""
import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from utils import CLASS_NAMES, IMAGE_EXTS, STAGING_DIR

CMIRD_NAMES = {0: "auto-rickshaw", 1: "bike", 2: "bus", 3: "car", 4: "crack",
               5: "manhole", 6: "person", 7: "pothole", 8: "speed bump", 9: "waterlogging"}
TO_OURS = {7: 0, 4: 1, 9: 2}


def prefix_of(stem):
    return stem.split("_")[0] if "_" in stem else stem[:6]


def group_of(stem):
    """Frames that look alike must stay together when splitting:
    video_07_frame_000123 -> 'video_07'; anomaly_vlcsnap-2025-02-19-17h08m40s964 -> 'anomaly_2025-02-19-17h'."""
    m = re.match(r"^video_(\d+)_", stem)
    if m:
        return f"video_{m.group(1)}"
    m = re.search(r"vlcsnap-(\d{4}-\d{2}-\d{2})-(\d{2})h", stem)
    if m:
        return f"anomaly_{m.group(1)}-{m.group(2)}h"
    return prefix_of(stem)


def find_labels_dir(src, det_dir):
    if det_dir:
        d = Path(det_dir)
        return d / "labels" if (d / "labels").is_dir() else d
    found = [p for p in src.rglob("labels") if p.is_dir() and "detection-object" in p.parts]
    if len(found) != 1:
        print("Expected exactly one detection-object/labels folder, found:", *found, sep="\n  ")
        raise SystemExit("Pass the right one with --det-dir (the detection-object folder).")
    return found[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder where CMIRD was extracted")
    ap.add_argument("--det-dir", default=None, help="the detection-object folder, if auto-detection fails")
    ap.add_argument("--dry-run", action="store_true", help="only report, write nothing")
    ap.add_argument("--keep-empty", action="store_true",
                    help="also keep images that contain none of our 3 classes (as background images)")
    args = ap.parse_args()

    src = Path(args.src)
    labels_dir = find_labels_dir(src, args.det_dir)
    images_dir = labels_dir.parent / "images"
    print(f"Using labels from: {labels_dir}")

    stem_lookup = None            # built only if the sibling images/ folder lookup fails
    stats = defaultdict(lambda: {"files": 0, "kept": 0, "ours": Counter(), "dropped": Counter()})
    names_by_prefix = defaultdict(list)
    to_write, bad_lines, no_image = [], 0, 0
    seen = {}                                   # image name -> (original split, converted lines)
    dup_pairs, dup_same, dup_diff, diff_examples = Counter(), 0, 0, []
    groups = defaultdict(lambda: {"images": 0, "ours": Counter()})

    for lp in sorted(labels_dir.rglob("*.txt")):
        split = lp.relative_to(labels_dir).parts[0] if len(lp.relative_to(labels_dir).parts) > 1 else ""
        pre = prefix_of(lp.stem)
        lines, dropped = [], Counter()
        for raw in lp.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = raw.split()
            if not parts:
                continue
            try:
                cid, vals = int(float(parts[0])), [float(v) for v in parts[1:]]
            except ValueError:
                bad_lines += 1
                continue
            if len(vals) != 4 or not all(0 <= v <= 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                bad_lines += 1
                continue
            if cid in TO_OURS:
                lines.append(f"{TO_OURS[cid]} " + " ".join(parts[1:]))
            else:
                dropped[CMIRD_NAMES.get(cid, f"unknown-{cid}")] += 1

        # the same image name seen again (e.g. in both train2017 and val2017) is a duplicate
        if lp.stem in seen:
            first_split, first_lines = seen[lp.stem]
            dup_pairs[(first_split, split)] += 1
            if sorted(first_lines) == sorted(lines):
                dup_same += 1
            else:
                dup_diff += 1
                if len(diff_examples) < 5:
                    diff_examples.append(lp.stem)
            continue
        seen[lp.stem] = (split, lines)

        names_by_prefix[pre].append(lp.stem)
        st = stats[pre]
        st["files"] += 1
        st["dropped"].update(dropped)
        for x in lines:
            st["ours"][CLASS_NAMES[int(x.split()[0])]] += 1

        if not lines and not args.keep_empty:
            continue
        st["kept"] += 1

        img = next((p for e in IMAGE_EXTS for p in [images_dir / split / (lp.stem + e)] if p.is_file()), None)
        if img is None:
            if stem_lookup is None:
                stem_lookup = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
            img = stem_lookup.get(lp.stem)
        if img is None:
            no_image += 1
            continue
        g = groups[group_of(lp.stem)]
        g["images"] += 1
        g["ours"].update(CLASS_NAMES[int(x.split()[0])] for x in lines)
        to_write.append((lp, img, split, pre, lines))

    print("\nPer file-name group (prefix):")
    for pre, st in sorted(stats.items()):
        print(f"  [{pre}] label files: {st['files']} | images that would be kept: {st['kept']}")
        print(f"      our classes : {dict(st['ours'])}")
        print(f"      dropped     : {dict(st['dropped'])}")
        n = sorted(names_by_prefix[pre])
        print(f"      sample names: {n[:4]} ... {n[len(n) // 2:len(n) // 2 + 2]} ... {n[-3:]}")
    total = Counter()
    for st in stats.values():
        total.update(st["ours"])
    print(f"\nTOTAL boxes in our 3 classes: {dict(total)}")
    print(f"Malformed label lines skipped: {bad_lines} | labels without an image: {no_image}")
    print(f"\nDuplicate image names removed: {dup_same + dup_diff} "
          f"(identical labels: {dup_same}, DIFFERENT labels: {dup_diff})")
    if dup_pairs:
        print("   original splits of the duplicate pairs (first seen -> second):", dict(dup_pairs))
    if diff_examples:
        print("   examples with different labels:", diff_examples)
    print(f"\nGroups (frames in one group must stay in one split): {len(groups)}")
    print(f"   {'group':<28}{'images':>7}{'pothole':>9}{'crack':>7}{'water':>7}")
    for name in sorted(groups):
        g = groups[name]
        print(f"   {name:<28}{g['images']:>7}{g['ours']['pothole']:>9}{g['ours']['crack']:>7}{g['ours']['waterlogging']:>7}")

    if args.dry_run:
        print("\nDry run only: nothing was written.")
        return

    img_out, lbl_out = STAGING_DIR / "images", STAGING_DIR / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    manifest = STAGING_DIR / "manifest_cmird.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "prefix", "group", "original_split", "pothole", "crack", "waterlogging"])
        for lp, img, split, pre, lines in to_write:
            new = "cmird_" + lp.stem
            shutil.copy2(img, img_out / (new + img.suffix.lower()))
            (lbl_out / (new + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            c = Counter(int(x.split()[0]) for x in lines)
            w.writerow([new, pre, group_of(lp.stem), split, c[0], c[1], c[2]])
    print(f"\nWrote {len(to_write)} images + labels to {STAGING_DIR}. Manifest: {manifest}")


if __name__ == "__main__":
    main()
