"""Leakage-safe train/val/test split: whole GROUPS go to one split.

Why: consecutive video frames are almost identical. If frame 2 of a video is in
training and frame 3 is in the test set, the test score is inflated.  Here every
image of a group (a video, or a screenshot session) goes to the same split.

How: many random group orderings are tried; each group is placed in the split that
keeps images AND pothole/crack/waterlogging boxes closest to the 70/15/15 targets.
The best of all trials is used.   Run (after prepare_* scripts):
    python src/split_by_group.py              # 70/15/15, 500 trials
    python src/split_by_group.py --overwrite  # rebuild dataset/ if it already exists
Images without a group (e.g. from other datasets) are treated as their own group.
"""
import argparse
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

from utils import CLASS_NAMES, DATASET_DIR, STAGING_DIR, find_images, write_data_yaml

SPLITS = ("train", "val", "test")
METRICS = ["images"] + CLASS_NAMES          # images, pothole, crack, waterlogging


def load_groups():
    """Return {group: {'files': [image paths], 'v': [images, pothole, crack, waterlogging]}}."""
    group_of = {}
    manifest = STAGING_DIR / "manifest_cmird.csv"
    if manifest.is_file():
        with open(manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                group_of[row["file"]] = row["group"]

    img_dir, lbl_dir = STAGING_DIR / "images", STAGING_DIR / "labels"
    if not img_dir.is_dir():
        raise SystemExit("dataset_staging/ not found. Run the prepare_* scripts first.")

    seen, groups, dups = set(), {}, 0
    for img in find_images(img_dir):
        h = hashlib.md5(img.read_bytes()).hexdigest()      # exact duplicates are dropped
        if h in seen:
            dups += 1
            continue
        seen.add(h)
        g = groups.setdefault(group_of.get(img.stem, img.stem), {"files": [], "v": [0] * len(METRICS)})
        g["files"].append(img)
        g["v"][0] += 1
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.is_file():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    g["v"][1 + int(line.split()[0])] += 1
    print(f"Exact duplicate images removed: {dups}")
    return groups


def assign(groups, ratios, rng):
    """One greedy attempt. Returns (group->split, per-split totals, error)."""
    names = list(groups)
    totals = [max(sum(groups[n]["v"][k] for n in names), 1) for k in range(len(METRICS))]
    target = {s: [ratios[s] * t for t in totals] for s in SPLITS}
    cur = {s: [0] * len(METRICS) for s in SPLITS}
    result = {}
    for n in sorted(names, key=lambda n: -groups[n]["v"][0] * rng.uniform(0.6, 1.4)):
        v = groups[n]["v"]
        def cost(s):
            return sum((((cur[s][k] + v[k] - target[s][k]) / totals[k]) ** 2
                        - ((cur[s][k] - target[s][k]) / totals[k]) ** 2) for k in range(len(METRICS)))
        best = min(SPLITS, key=cost)
        result[n] = best
        for k in range(len(METRICS)):
            cur[best][k] += v[k]
    err = sum(((cur[s][k] - target[s][k]) / totals[k]) ** 2 for s in SPLITS for k in range(len(METRICS)))
    err += 10 * sum(1 for s in SPLITS for k in range(1, len(METRICS)) if cur[s][k] == 0)  # no empty class
    return result, cur, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)       # the rest is test
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="delete an existing dataset/ first")
    ap.add_argument("--dry-run", action="store_true", help="show the split, copy nothing")
    args = ap.parse_args()

    groups = load_groups()
    ratios = {"train": args.train, "val": args.val, "test": 1 - args.train - args.val}
    rng = random.Random(args.seed)
    best = min((assign(groups, ratios, rng) for _ in range(args.trials)), key=lambda r: r[2])
    result, totals, err = best

    print(f"\nBest of {args.trials} trials (seed {args.seed}, error {err:.4f}). Groups: {len(groups)}")
    grand = [sum(totals[s][k] for s in SPLITS) for k in range(len(METRICS))]
    print(f"{'split':<7}{'groups':>7}" + "".join(f"{m:>14}" for m in METRICS))
    for s in SPLITS:
        n_groups = sum(1 for g in result.values() if g == s)
        cells = "".join(f"{totals[s][k]:>8} ({100 * totals[s][k] / max(grand[k], 1):>3.0f}%)" for k in range(len(METRICS)))
        print(f"{s:<7}{n_groups:>7}{cells}")
    for s in ("val", "test"):
        print(f"{s} groups: {sorted(n for n, g in result.items() if g == s)}")
    small = [(s, METRICS[k]) for s in ("val", "test") for k in range(1, len(METRICS)) if totals[s][k] < 30]
    if small:
        print("WARNING: very few boxes for", small, "- metrics for these classes will be unreliable.")

    if args.dry_run:
        print("\nDry run only: nothing was copied.")
        return
    if DATASET_DIR.exists():
        if not args.overwrite:
            raise SystemExit("dataset/ already exists. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(DATASET_DIR)

    for name, g in groups.items():
        split = result[name]
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in g["files"]:
            shutil.copy2(img, DATASET_DIR / "images" / split / img.name)
            lbl = STAGING_DIR / "labels" / (img.stem + ".txt")
            if lbl.is_file():
                shutil.copy2(lbl, DATASET_DIR / "labels" / split / lbl.name)
    with open(DATASET_DIR / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "split"] + METRICS)
        for name in sorted(groups):
            w.writerow([name, result[name]] + groups[name]["v"])
    print(f"\nWrote dataset/ and {write_data_yaml().name}. Next: python src/check_dataset.py")


if __name__ == "__main__":
    main()
