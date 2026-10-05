"""Pick one confidence threshold PER CLASS from the VALIDATION split, then check it on the TEST split.

Why: one threshold for all classes is a compromise. Waterlogging may need a higher threshold (fewer
false alarms) while cracks and potholes may need a lower one (to find more of them). The best threshold
for a class is the one that gives it the highest F1 score (balance of precision and recall).
Thresholds are chosen on the validation split only; the test split is just used to report the result,
so the test numbers stay honest.

Run (from the project folder, takes a few minutes on a CPU):
    python src/tune_thresholds.py --imgsz 960
Writes weights/class_thresholds.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from utils import DEFAULT_WEIGHTS, WEIGHTS_DIR, pick_device, write_data_yaml

NEEDED = ("px", "f1_curve", "p_curve", "r_curve", "ap_class_index")


def best_thresholds(px, f1_curve, class_index, names):
    """For each class: (threshold that maximises F1, that F1). f1_curve has one row per class present."""
    out = {}
    for row, cid in enumerate(class_index):
        k = int(np.argmax(f1_curve[row]))
        out[names[int(cid)]] = (float(px[k]), float(f1_curve[row][k]))
    return out


def values_at(px, curves, class_index, names, thresholds):
    """Precision, recall, F1 of each class when its confidence threshold is thresholds[class]."""
    p_curve, r_curve = curves
    res = {}
    for row, cid in enumerate(class_index):
        name = names[int(cid)]
        k = int(np.argmin(np.abs(px - thresholds[name])))
        p, r = float(p_curve[row][k]), float(r_curve[row][k])
        res[name] = (p, r, 2 * p * r / (p + r) if (p + r) else 0.0)
    return res


def run_val(weights, data, split, imgsz, device):
    from ultralytics import YOLO
    box = YOLO(weights).val(data=data, split=split, imgsz=imgsz, device=device, plots=False, verbose=False).box
    missing = [a for a in NEEDED if not hasattr(box, a)]
    if missing:
        raise SystemExit(f"This Ultralytics version does not expose {missing}. Send me this message.")
    return box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--baseline", type=float, default=0.15, help="the single threshold to compare against")
    args = ap.parse_args()
    if not Path(args.weights).is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    device = pick_device() if args.device == "auto" else args.device
    data = str(write_data_yaml())
    names = None

    val = run_val(args.weights, data, "val", args.imgsz, device)
    test = run_val(args.weights, data, "test", args.imgsz, device)
    from ultralytics import YOLO
    names = YOLO(args.weights).names

    best = best_thresholds(val.px, val.f1_curve, val.ap_class_index, names)
    thr = {n: t for n, (t, _) in best.items()}
    tuned = values_at(test.px, (test.p_curve, test.r_curve), test.ap_class_index, names, thr)
    base = values_at(test.px, (test.p_curve, test.r_curve), test.ap_class_index, names,
                     {n: args.baseline for n in thr})

    print(f"\nThresholds chosen on the VALIDATION split, results on the TEST split")
    print(f"{'class':<14}{'val-best thr':>13}{'val F1':>8} |{'test P':>8}{'R':>7}{'F1':>7} |"
          f"{f'P@{args.baseline}':>9}{'R':>7}{'F1':>7}")
    for n in thr:
        t, f = best[n]
        (p, r, f1), (bp, br, bf) = tuned[n], base[n]
        print(f"{n:<14}{t:>13.2f}{f:>8.3f} |{p:>8.3f}{r:>7.3f}{f1:>7.3f} |{bp:>9.3f}{br:>7.3f}{bf:>7.3f}")

    out = WEIGHTS_DIR / "class_thresholds.json"
    out.write_text(json.dumps({"thresholds": thr, "chosen_on": "validation split", "imgsz": args.imgsz},
                              indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    print("Left block: each class at its own threshold. Right block: every class at the single baseline threshold.")


if __name__ == "__main__":
    main()
