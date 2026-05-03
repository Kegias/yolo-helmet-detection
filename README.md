# YOLOv8 Helmet Detection using Transfer Learning

## Overview

This project implements an object detection model for detecting helmet usage using YOLOv8. The model identifies two classes: **With Helmet** and **Without Helmet**. It was trained using transfer learning on a Roboflow object detection dataset with bounding-box annotations.

The goal of this project was to build an end-to-end computer vision pipeline covering dataset download, YOLOv8 training, validation, prediction, and result documentation.

## Dataset

- Source: Roboflow Universe
- Dataset: Helmet Detection_YOLOv8
- Type: Object Detection
- Classes:
  - With Helmet
  - Without Helmet
- Format: YOLOv8
- Version: 3

The dataset was downloaded using the Roboflow API. The API key is not stored in this repository.

## Model

The project uses the pretrained YOLOv8 nano model:

```python
model.train(
    data="/content/Helmet-Detection_YOLOv8-3/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    name="helmet_detection_yolov8"
)
```

## Results

Validation performance:

| Metric | Value |
|---|---:|
| Precision | 85.50% |
| Recall | 90.71% |
| mAP50 | 93.12% |
| mAP50-95 | 58.30% |

## Sample Predictions

Sample model predictions are available in:

```text
results/predictions/
```

Training plots and evaluation outputs are available in:

```text
results/training/
```

## Project Structure

```text
yolo-object-detection-roboflow/
├── notebooks/
│   └── YOLOv8_Helmet_Detection_Object_Detection.ipynb
├── models/
│   └── best.pt
├── results/
│   ├── training/
│   └── predictions/
├── src/
│   └── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run prediction using the trained model:

```bash
python src/predict.py
```

## Key Learnings

- Built an object detection pipeline using YOLOv8.
- Used transfer learning with pretrained YOLO weights.
- Worked with bounding-box annotations in YOLO format.
- Evaluated model performance using precision, recall, mAP50, and mAP50-95.
- Generated visual predictions with bounding boxes for helmet detection.

## Future Improvements

- Train with larger YOLO variants such as YOLOv8s or YOLOv8m.
- Improve detection under low-light and crowded conditions.
- Tune confidence threshold and IoU threshold.
- Deploy the model using Streamlit, Flask, or FastAPI.