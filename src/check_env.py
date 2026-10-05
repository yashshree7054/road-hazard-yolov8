"""Step 1: check that Python, PyTorch, GPU and Ultralytics are working.

Run:   python src/check_env.py
Add    --run-test   to also download yolov8n.pt and run it on a sample image
       (needs internet the first time).
"""
import argparse
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-test", action="store_true",
                        help="run the pretrained YOLOv8n on a sample image")
    args = parser.parse_args()

    print(f"Python      : {sys.version.split()[0]}")
    print(f"System      : {platform.system()} {platform.release()}")

    import torch
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA GPU    : {'YES - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no (training will use CPU)'}")

    import ultralytics
    print(f"Ultralytics : {ultralytics.__version__}")

    if args.run_test:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # downloaded automatically on first use
        results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
        # This model knows the 80 COCO classes (person, bus, ...), NOT our road hazards.
        print("Test detections (COCO classes):",
              [model.names[int(c)] for c in results[0].boxes.cls])


if __name__ == "__main__":
    main()
