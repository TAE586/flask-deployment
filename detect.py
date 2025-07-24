from ultralytics import YOLO

model = YOLO('best.pt')

results = model("test/002.jpg", conf=0.05)
results[0].show()