"""Reusable detection functions used by the command-line scripts AND the Streamlit app."""
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2


def find_ffmpeg():
    """Return a path to ffmpeg: one on the system PATH, else the one bundled with imageio-ffmpeg, else None."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def video_duration_seconds(path):
    """Length of a video in seconds. None if it cannot be opened; 0.0 if the length is unknown."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or fps < 1 or not frames or frames < 1:
        return 0.0
    return frames / fps


def load_model(weights):
    from ultralytics import YOLO
    if str(weights).endswith(".pt") and not Path(weights).is_file() and Path(weights).name != "yolov8n.pt":
        raise FileNotFoundError(f"Weights not found: {weights}. Train a model first or pass --weights.")
    return YOLO(str(weights))


def detect_frame(model, frame_bgr, conf=0.25, imgsz=640):
    """Run detection on one BGR image (NumPy array). Returns (annotated_bgr, result)."""
    result = model.predict(frame_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    return result.plot(), result      # plot() draws boxes + class names + confidence


def result_to_rows(model, result):
    """List of dicts (class, confidence, box) - handy for tables and printing."""
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(float(v)) for v in box.xyxy[0])
        rows.append({"class": model.names[int(box.cls[0])], "confidence": round(float(box.conf[0]), 3),
                     "box (x1,y1,x2,y2)": f"{x1},{y1},{x2},{y2}"})
    return rows


def _open_writer(path, fps, size):
    """Try a few codecs; return (writer, codec). Raises if none work."""
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not open a video writer. Try a different output extension (.avi).")


def process_video(model, src, dst, conf=0.25, imgsz=640, stride=1, progress=None):
    """Detect hazards in a video and write an annotated video.

    stride=1 : detect on every frame (most accurate, slowest)
    stride=N : run the model on every N-th frame only; the frames in between
               reuse the last detections so the output video keeps its full length.
    Returns a dict with a summary (frames, detections per class, output path).
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}. The file may be damaged or use an unsupported codec.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0           # some files report 0 FPS
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw_path = dst.with_name(dst.stem + "_raw.mp4")
    writer = _open_writer(raw_path, fps, (width, height))

    counts, last, idx = Counter(), None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, stride) == 0:
            annotated, last = detect_frame(model, frame, conf, imgsz)
            for c in last.boxes.cls:
                counts[model.names[int(c)]] += 1
        else:
            annotated = last.plot(img=frame) if last is not None else frame
        writer.write(annotated)
        idx += 1
        if progress and total:
            progress(min(idx / total, 1.0))
    cap.release()
    writer.release()
    if idx == 0:
        raise RuntimeError("The video contained no readable frames.")

    # Browsers cannot play 'mp4v'. If ffmpeg is installed, convert to H.264.
    playable = False
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        proc = subprocess.run([ffmpeg, "-y", "-i", str(raw_path), "-vcodec", "libx264",
                               "-pix_fmt", "yuv420p", str(dst)], capture_output=True, text=True)
        if proc.returncode == 0:
            raw_path.unlink()
            playable = True
        else:
            print("ffmpeg conversion failed:\n", proc.stderr[-500:])
            raw_path.replace(dst)
    else:
        raw_path.replace(dst)
    return {"frames": idx, "fps": fps, "detections": dict(counts), "output": str(dst), "browser_playable": playable}
