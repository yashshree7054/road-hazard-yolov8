"""Step 5: fine-tune a pretrained YOLOv8n on the road-hazard dataset.

    python src/train.py --epochs 3 --name smoke_test     # quick pipeline check
    python src/train.py --epochs 50 --name yolov8n_v1    # real run

Beginner-friendly meanings:
  epochs  : how many times the model sees the whole training set
  imgsz   : images are resized to imgsz x imgsz pixels before training
  batch   : how many images are processed at once (lower it if memory runs out)
  patience: stop early if validation results do not improve for this many epochs
"""
import argparse
import shutil
import time

from utils import (CLASS_NAMES, DATASET_DIR, DEFAULT_WEIGHTS, RUNS_DIR, hardware_description,
                   log_experiment, pick_device, write_data_yaml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained starting weights")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2, help="data-loading processes (use 0 on Windows if errors)")
    ap.add_argument("--name", default="yolov8n_v1", help="run name; results go to runs/<name>")
    ap.add_argument("--device", default="auto", help="auto, cpu, 0 (first GPU) or mps")
    args = ap.parse_args()

    if not (DATASET_DIR / "images" / "train").is_dir():
        raise SystemExit("dataset/ is missing. Run prepare_* and split_dataset.py first.")

    from ultralytics import YOLO

    data_yaml = write_data_yaml()
    device = pick_device() if args.device == "auto" else args.device
    print(f"Training on device: {device}")

    model = YOLO(args.model)            # downloads yolov8n.pt the first time
    start = time.time()
    model.train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, workers=args.workers, device=device,
        project=str(RUNS_DIR), name=args.name, seed=0,
    )
    minutes = (time.time() - start) / 60

    save_dir = model.trainer.save_dir
    best = save_dir / "weights" / "best.pt"
    DEFAULT_WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy2(best, DEFAULT_WEIGHTS)
    print(f"\nBest weights copied to {DEFAULT_WEIGHTS}")

    # Validation-set metrics of best.pt for the experiment log
    m = YOLO(str(best)).val(data=str(data_yaml), split="val", imgsz=args.imgsz, device=device, verbose=False).box
    f1 = 2 * m.mp * m.mr / (m.mp + m.mr) if (m.mp + m.mr) else 0.0
    log_experiment({
        "run_name": args.name, "model": args.model, "num_classes": len(CLASS_NAMES),
        "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch,
        "split": "train/val/test from dataset/ (seed 42)",
        "precision": round(float(m.mp), 4), "recall": round(float(m.mr), 4), "f1": round(float(f1), 4),
        "map50": round(float(m.map50), 4), "map50_95": round(float(m.map), 4),
        "train_minutes": round(minutes, 1), "hardware": hardware_description(),
        "metrics_split": "val",
    })


if __name__ == "__main__":      # this guard is required on Windows
    main()
