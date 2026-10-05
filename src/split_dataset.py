"""Step 3: remove duplicates and split the staged data into train / val / test.

    python src/split_dataset.py                       # 70/15/15 split
    python src/split_dataset.py --group-near-duplicates   # slower, safer against leakage

Why the split is done per source: RDD2022 images (prefix rdd_) vastly outnumber
waterlogging images (prefix water_). Splitting each source separately guarantees
that every split contains some waterlogging images.

Data leakage: if near-identical photos end up in both train and test, the test
scores look better than they really are. Exact copies are always removed.
--group-near-duplicates also keeps visually near-identical images together in
the SAME split (uses a small image fingerprint; slow on very large sets).
"""
import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from utils import DATASET_DIR, STAGING_DIR, find_images, write_data_yaml


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def average_hash(path):
    """64-bit fingerprint: shrink to 8x8 grey, mark pixels brighter than the average."""
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((8, 8)), dtype=np.float32)
    return (small > small.mean()).flatten()


def group_near_duplicates(paths, max_distance=4):
    """Union-find over images whose fingerprints differ by <= max_distance bits."""
    hashes = np.stack([average_hash(p) for p in paths])
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths)):
        dist = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.nonzero(dist <= max_distance)[0]:
            parent[find(i)] = find(i + 1 + int(j))
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[find(i)].append(p)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)   # the rest becomes test
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-near-duplicates", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing dataset/ first")
    args = ap.parse_args()

    img_dir, lbl_dir = STAGING_DIR / "images", STAGING_DIR / "labels"
    if not img_dir.is_dir():
        raise SystemExit("dataset_staging/ not found. Run the prepare_* scripts first.")
    if DATASET_DIR.exists():
        if not args.overwrite:
            raise SystemExit("dataset/ already exists. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(DATASET_DIR)

    images = find_images(img_dir)
    print(f"Staged images: {len(images)}")

    # 1) remove exact duplicates
    seen, unique, dup = {}, [], 0
    for p in images:
        h = md5(p)
        if h in seen:
            dup += 1
        else:
            seen[h] = p
            unique.append(p)
    print(f"Exact duplicates removed: {dup}")

    # 2) split inside each source (rdd_ / water_ / ...)
    by_source = defaultdict(list)
    for p in unique:
        by_source[p.name.split("_")[0]].append(p)

    rng = random.Random(args.seed)
    assignment = {}
    for source, paths in sorted(by_source.items()):
        units = group_near_duplicates(paths) if args.group_near_duplicates else [[p] for p in paths]
        rng.shuffle(units)
        total = len(paths)
        n_train, n_val = args.train * total, (args.train + args.val) * total
        count = 0
        for unit in units:
            split = "train" if count < n_train else "val" if count < n_val else "test"
            for p in unit:
                assignment[p] = split
            count += len(unit)

    # 3) copy files
    counts = defaultdict(int)
    for p, split in assignment.items():
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, DATASET_DIR / "images" / split / p.name)
        lbl = lbl_dir / (p.stem + ".txt")
        if lbl.is_file():
            shutil.copy2(lbl, DATASET_DIR / "labels" / split / lbl.name)
        counts[(p.name.split("_")[0], split)] += 1

    for (source, split), n in sorted(counts.items()):
        print(f"  {source:<8} {split:<6} {n}")
    print(f"Wrote {write_data_yaml().name}. Next: python src/check_dataset.py")


if __name__ == "__main__":
    main()
