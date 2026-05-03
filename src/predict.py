import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Run YOLOv8 helmet detection on an image or folder.")
parser.add_argument("--source", required=True, help="Path to an image or folder of images.")
parser.add_argument("--model", default="models/best.pt", help="Path to trained YOLOv8 model.")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")

args = parser.parse_args()

model = YOLO(args.model)

results = model.predict(
    source=args.source,
    conf=args.conf,
    save=True
)

print("Prediction completed.")
print("Results saved in the runs/detect folder.")