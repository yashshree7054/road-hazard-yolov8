"""Step 6: evaluate a trained model on the held-out TEST split.

    python src/evaluate.py                      # uses weights/best.pt on the test split

Prints Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 overall and per class.
Plots (confusion matrix, PR curve) are saved in runs/eval_<split>/.
Only numbers produced by this run are reported - nothing is estimated.
"""
import argparse

from utils import DEFAULT_WEIGHTS, RUNS_DIR, log_experiment, pick_device, write_data_yaml, hardware_description, CLASS_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from pathlib import Path
    if not Path(args.weights).is_file():
        raise SystemExit(f"Weights not found: {args.weights}. Train first (src/train.py).")

    from ultralytics import YOLO

    device = pick_device() if args.device == "auto" else args.device
    metrics = YOLO(args.weights).val(
        data=str(write_data_yaml()), split=args.split, imgsz=args.imgsz, device=device,
        plots=True, project=str(RUNS_DIR), name=f"eval_{args.split}", exist_ok=True,
    )
    b = metrics.box
    f1 = lambda p, r: 2 * p * r / (p + r) if (p + r) else 0.0

    print(f"\n=== Results on the '{args.split}' split ===")
    print(f"{'class':<14}{'precision':>10}{'recall':>9}{'F1':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for i, cls_id in enumerate(b.ap_class_index):
        print(f"{metrics.names[int(cls_id)]:<14}{b.p[i]:>10.3f}{b.r[i]:>9.3f}{f1(b.p[i], b.r[i]):>8.3f}"
              f"{b.ap50[i]:>9.3f}{b.ap[i]:>10.3f}")
    print(f"{'ALL':<14}{b.mp:>10.3f}{b.mr:>9.3f}{f1(b.mp, b.mr):>8.3f}{b.map50:>9.3f}{b.map:>10.3f}")
    print("(A class missing from the table has no examples in this split.)")
    print(f"Confusion matrix and curves: runs/eval_{args.split}/")

    log_experiment({
        "run_name": f"eval_{args.split}", "model": args.weights, "num_classes": len(CLASS_NAMES),
        "imgsz": args.imgsz, "epochs": "", "batch": "", "split": args.split,
        "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4),
        "f1": round(float(f1(b.mp, b.mr)), 4), "map50": round(float(b.map50), 4),
        "map50_95": round(float(b.map), 4), "train_minutes": "", "hardware": hardware_description(),
        "metrics_split": args.split,
    })


if __name__ == "__main__":
    main()
