import os
import shutil
from ultralytics import YOLO

def main_model() -> None:
    '''Call all necessary for training model from dataset.          

    Return:
        No data. Save all training and validation results in `data/external/` directory.
    '''

    weights_path = "data/external/yolo_training_run_weights/weights/best.pt"
    conf = 0.25

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
            
            #augment = True,
            #degrees = 10,
            #scale = 0.5,
            #shear = 2,
            #flipud = 0.5,
            #fliplr = 0.5,
            
            patience = 4,  # early stopping
            #cos_lr = True   # lr adjust
        )

        metrics = model.val(conf = conf)

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
        results = model("assets/" + img, conf = conf)
        results[0].show()