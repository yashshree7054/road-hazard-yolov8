"""Step 7: detect hazards in one image or a folder of images.

    python src/detect_image.py --source path/to/road.jpg
    python src/detect_image.py --source path/to/folder --conf 0.4
Results are saved in outputs/images/.
"""
import argparse
from pathlib import Path

import cv2

from inference import detect_frame, load_model, result_to_rows
from utils import DEFAULT_WEIGHTS, OUTPUT_DIR, find_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="an image file or a folder of images")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (0-1)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(OUTPUT_DIR / "images"))
    args = ap.parse_args()

    src = Path(args.source)
    files = find_images(src) if src.is_dir() else [src]
    if not files or not all(f.is_file() for f in files):
        raise SystemExit(f"No image found at {src}")

    model = load_model(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"[skip] cannot read {f}")
            continue
        annotated, result = detect_frame(model, frame, args.conf, args.imgsz)
        dst = out_dir / f"{f.stem}_detected.jpg"
        cv2.imwrite(str(dst), annotated)
        rows = result_to_rows(model, result)
        print(f"{f.name}: {len(rows)} detection(s) -> {dst}")
        for r in rows:
            print(f"    {r['class']:<13} conf={r['confidence']}")


if __name__ == "__main__":
    main()
