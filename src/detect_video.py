"""Step 8: detect hazards in a video.

    python src/detect_video.py --source path/to/road.mp4
    python src/detect_video.py --source road.mp4 --stride 3     # faster: model runs on every 3rd frame
Result is saved in outputs/videos/.
"""
import argparse
from pathlib import Path

from inference import load_model, process_video
from utils import DEFAULT_WEIGHTS, OUTPUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=1, help="run detection on every N-th frame")
    ap.add_argument("--out", default=None, help="output file (default: outputs/videos/<name>_detected.mp4)")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_file():
        raise SystemExit(f"Video not found: {src}")
    dst = Path(args.out) if args.out else OUTPUT_DIR / "videos" / f"{src.stem}_detected.mp4"

    model = load_model(args.weights)
    last_pct = [-1]

    def show(p):
        pct = int(p * 100)
        if pct // 10 != last_pct[0] // 10:
            print(f"  {pct}%")
        last_pct[0] = pct

    summary = process_video(model, src, dst, args.conf, args.imgsz, args.stride, show)
    print(f"Done: {summary['frames']} frames -> {summary['output']}")
    print("Detections per class (counted on processed frames):", summary["detections"])
    if not summary["browser_playable"]:
        print("Note: install ffmpeg if the output does not play in your browser (VLC plays it anyway).")


if __name__ == "__main__":
    main()
