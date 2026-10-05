"""Web interface for the road-hazard detector.

Run from the project folder (the one that contains src/ and weights/):
    streamlit run src/app.py
Then open the address it prints (normally http://localhost:8501).
"""
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from inference import detect_frame, load_model, process_video, result_to_rows, video_duration_seconds
from utils import DEFAULT_WEIGHTS

MAX_VIDEO_SECONDS = 20   # the page runs on a CPU, so long videos would take too long

# Real results measured on the held-out TEST split (254 images) for this model.
# If you replace weights/best.pt with a different run, update this text.
MODEL_NOTE = """
**Model:** YOLOv8n fine-tuned on CMIRD (train 1,102 / val 283 / test 254 images, split by video).
Run `yolov8n_cmird_v2_imgsz960`: 960 px images, 50 epochs.

**Test-split results:** precision 0.346, recall 0.184, mAP@0.5 0.177, mAP@0.5:0.95 0.064.

| Class | mAP@0.5 | What this means |
|---|---|---|
| waterlogging | 0.353 | works best |
| pothole | 0.137 | finds some, misses many |
| crack | 0.040 | finds almost none |

Confidence scores from this model are low, so the slider starts at 0.15. Lowering it shows more
detections but also more wrong ones.
"""

st.set_page_config(page_title="Road Hazard Detection", layout="wide")
st.title("Smart Road Safety Monitoring System")
st.caption("YOLOv8 detection of potholes, cracks and waterlogging on road images and videos. "
           "Uploaded files are processed temporarily and are not kept by this app.")

# ---------------- sidebar settings ----------------
st.sidebar.header("Settings")
weights = st.sidebar.text_input("Model weights", str(DEFAULT_WEIGHTS))
if not Path(weights).is_file():
    st.warning(f"No model file found at: {weights}. Put your trained best.pt in the weights folder "
               "or change the path in the sidebar.")
    st.stop()


@st.cache_resource
def get_model(path):
    return load_model(path)


model = get_model(weights)

# Use the image size the model was trained with as the default
trained_imgsz = (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 960)
sizes = sorted({640, 800, 960, 1280, int(trained_imgsz)})

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.90, 0.15, 0.05,
                         help="Detections below this confidence are hidden.")
imgsz = st.sidebar.selectbox("Image size (pixels)", sizes, index=sizes.index(int(trained_imgsz)),
                             help="Use the size the model was trained at (the default).")
stride = st.sidebar.slider("Video: run the model on every N-th frame", 1, 10, 3,
                           help="Higher = faster but may miss short-lived hazards.")

with st.expander("About this model and its limits"):
    st.markdown(MODEL_NOTE)


def show_counts(counts):
    """One number per class: how many detections of each."""
    names = [model.names[i] for i in sorted(model.names)]
    for col, name in zip(st.columns(len(names)), names):
        col.metric(name, counts.get(name, 0))


tab_img, tab_vid = st.tabs(["Image", "Video"])

# ---------------- image tab ----------------
with tab_img:
    up = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="img")
    if up is not None:
        frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("This file could not be read as an image.")
        else:
            annotated, result = detect_frame(model, frame, conf, imgsz)
            rows = result_to_rows(model, result)
            left, right = st.columns(2)
            left.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Input")
            right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections")
            show_counts(Counter(r["class"] for r in rows))
            if rows:
                st.table(rows)
            else:
                st.info("No detections above this threshold. Try lowering the confidence slider.")
            ok, png = cv2.imencode(".png", annotated)
            if ok:
                st.download_button("Download result image", png.tobytes(),
                                   f"{Path(up.name).stem}_detected.png", "image/png")

# ---------------- video tab ----------------
with tab_vid:
    st.caption(f"Processing runs on a CPU, so videos are limited to {MAX_VIDEO_SECONDS} seconds.")
    vid = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv", "m4v", "webm"], key="vid")
    if vid is not None and st.button("Run detection on video"):
        with tempfile.TemporaryDirectory() as tmp:       # everything here is deleted afterwards
            src = Path(tmp) / ("input" + Path(vid.name).suffix)
            src.write_bytes(vid.getvalue())
            seconds = video_duration_seconds(src)
            if seconds is None:
                st.error("This video could not be opened. Try an MP4 file.")
            elif seconds > MAX_VIDEO_SECONDS:
                st.error(f"This video is {seconds:.0f} seconds long. Please upload a clip of at most "
                         f"{MAX_VIDEO_SECONDS} seconds.")
            else:
                dst = Path(tmp) / f"{Path(vid.name).stem}_detected.mp4"
                bar = st.progress(0.0)
                try:
                    summary = process_video(model, src, dst, conf, imgsz, stride, bar.progress)
                except RuntimeError as err:
                    st.error(str(err))
                else:
                    st.success(f"Processed {summary['frames']} frames.")
                    show_counts(Counter(summary["detections"]))
                    data = Path(summary["output"]).read_bytes()
                    if summary["browser_playable"]:
                        st.video(data)
                    else:
                        st.info("This video could not be converted for the browser. Download it and open "
                                "it with VLC.")
                    st.download_button("Download video", data, Path(summary["output"]).name, "video/mp4")
