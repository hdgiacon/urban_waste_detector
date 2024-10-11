import os
import shutil
from ultralytics import YOLO

def main_model() -> None:
    '''Call all necessary for training model from dataset.          

    Return:
        No data. Save all training and validation results in `data/external/` directory.
    '''

    weights_path = "data/external/yolo_trained_model.pt"

    model = YOLO("data/external/yolo11s.pt")

    train_results = model.train(
        data = "data/processed/data.yaml",
        epochs = 50,
        imgsz = 640,
        device = "0",
        project = "data/external",
        name = "yolo_training_run_weights",
    )

    metrics = model.val()

    results = model("assets/img_test.jpg")
    results[0].show()

    # Exportar o modelo para ONNX
    #model.export(format="onnx", project = "data/external/yolo_model")

    # Exportar o modelo para TFLite
    model.export(format="tflite", project = "data/external/yolo_model")

    n_weights_path = "yolo11n.pt"
    destination_path = "data/raw/yolo11n.pt"

    if os.path.exists(n_weights_path):
        shutil.move(n_weights_path, destination_path)
    
    else:
        print(f"O arquivo {n_weights_path} não foi encontrado.")