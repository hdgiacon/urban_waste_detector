from ultralytics import YOLO

model = YOLO("yolo11s.pt")

train_results = model.train(
    data = "data/processed/data.yaml",
    epochs = 50,
    imgsz = 640,
    device = "0",
)

metrics = model.val()

results = model("assets/img_test.jpg")
results[0].show()

path = model.export(format = "onnx")