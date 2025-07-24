import yaml

data_path = "D:/yolov11_newcastle/dataset/data.yaml"

with open(data_path, "r") as f:
    data = yaml.safe_load(f)
print("ใช้ data.yaml ที่ path:", data_path)
print("จำนวนคลาส nc =", data["nc"])
print("ชื่อคลาส =", data["names"])

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data=data_path, epochs=5, imgsz=640, device="0", workers=0)
