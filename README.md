# Smart Road Safety Monitoring System Using YOLOv8
**Pothole, crack and waterlogging detection on road images and videos**

Computer Vision · Deep Learning · Object Detection

A YOLOv8n model fine-tuned on the Custom Multitask Indian Road Dataset (CMIRD) detects three road hazards
(pothole, crack, waterlogging) and draws boxes with class names and confidence scores. A Streamlit page
lets you upload an image or a short video and view the result.

> **This is a student baseline, not a safety system.** Results below are measured on a held-out test split
> and are modest. Crack detection in particular works poorly.

## Results (held-out test split, 254 images, split by video so no frames leak between splits)
| Run | Image size | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| v1 | 640 | 0.218 | 0.208 | 0.213 | 0.148 | 0.050 |
| **v2 (used by the app)** | 960 | 0.346 | 0.184 | 0.240 | 0.177 | 0.064 |

Per class, run v2: waterlogging mAP@0.5 0.353, pothole 0.137, crack 0.040 (recall about 1%).
Both runs: YOLOv8n, 50 epochs, batch 16, Tesla T4 (Google Colab). Run v2 was resumed once from its last
checkpoint at epoch 40 after a Colab disconnect. Each number comes from a single run.

## Dataset
CMIRD (Kaggle; two cities in southern India, dashcam video frames), licence **CC BY-NC-SA 4.0**
(non-commercial, attribution, share-alike). Only the detection labels were used. CMIRD classes were mapped as:
pothole -> 0, crack -> 1, waterlogging -> 2; vehicles, persons, manholes and speed bumps were dropped.
540 duplicate images (the same file in both original splits) were removed, and the remaining frames were
split by video into train 1,102 / val 283 / test 254 images.

## Limitations
- Crack and pothole labels in the dataset overlap in meaning, and cracks are thin and low contrast.
- Most training cracks come from a different source than the test cracks.
- Small training set; one run per setting; waterlogging includes wet patches and puddles.
- The page runs on a CPU, so videos are limited to 20 seconds.

## Run locally
```
pip install -r requirements.txt
python -m streamlit run src/app.py
```
Scripts in `src/`: `prepare_cmird.py`, `split_by_group.py`, `check_dataset.py`, `train.py`, `evaluate.py`,
`detect_image.py`, `detect_video.py`, `app.py`. Training was done on Google Colab (notebook not included here).

## References (verify before submission)
- Jocher, G., Chaurasia, A., Qiu, J. *Ultralytics YOLOv8* (2023), https://github.com/ultralytics/ultralytics (AGPL-3.0).
- Custom Multitask Indian Road Dataset (CMIRD), Kaggle, CC BY-NC-SA 4.0; and the associated paper
  "Safety-aware transformer-enhanced unified multi-task perception for road anomaly detection, drivable area
  segmentation and lane estimation" (PubMed 42410140).
