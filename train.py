"""
Train the custom floating-plastic detector.

Dataset structure expected:

dataset/
    images/
        train/
        val/
    labels/
        train/
        val/

Create dataset/data.yaml such as:

path: /absolute/path/to/dataset
train: images/train
val: images/val

names:
  0: plastic

Then run:

python train.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="runs",
    name="marine_plastic"
)

print("Training completed.")
print("Copy the trained best.pt into weights/best.pt")
