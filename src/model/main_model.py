import os
import shutil
from ultralytics import YOLO

def main_model() -> None:
    '''Call all necessary for training model from dataset.          

    Return:
        No data. Save all training and validation results in `data/external/` directory.
    '''

    weights_path = "data/external/yolo_training_run_weights/weights/best.pt"

    if os.path.exists(weights_path):
        model = YOLO(weights_path)

    else:
        model = YOLO("data/external/yolo11s.pt")

        train_results = model.train(
            data = "data/processed/data.yaml",
            epochs = 50,
            imgsz = 640,
            device = "0",
            project = "data/external",
            name = "yolo_training_run_weights",
            #augment = True,  # Ativa o Data Augmentation
            #degrees = 10,  # Rotação de até 10 graus
            #scale = 0.5,  # Zoom de até 50%
            #shear = 2,    # Shear (distorsão) de até 2 graus
            #flipud = 0.5,  # Flip vertical com probabilidade de 50%
            #fliplr = 0.5,  # Flip horizontal com probabilidade de 50%
            #patience = 5,  # early stopping
            #cos_lr = True   # lr adjust
        )

        metrics = model.val(conf = 0.25)

        model.export(
            format = "tflite",
            imgsz = 640, 
            half = True, 
            #int8 = True, 
            batch = 64
        )

        n_weights_path = "yolo11n.pt"
        destination_path = "data/raw/yolo11n.pt"

        if os.path.exists(n_weights_path):
            shutil.move(n_weights_path, destination_path)
        
        else:
            print(f"O arquivo {n_weights_path} não foi encontrado.")

        calibration_img_path = "calibration_image_sample_data_20x128x128x3_float32.npy"
        destination_path = "data/external/calibration_image_sample_data_20x128x128x3_float32.npy"

        if os.path.exists(calibration_img_path):
            shutil.move(calibration_img_path, destination_path)
        
        else:
            print(f"O arquivo {calibration_img_path} não foi encontrado.")


    for img in os.listdir('assets/'):
        results = model("assets/" + img, conf = 0.25)
        results[0].show()