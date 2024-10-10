import sys
import json

from .convert_coco_to_yolo import ConvertCocoToYolo
from .train_valid_data_files import TrainValidDataFiles

def main_features():
    json_file_path = 'data/raw/annotations.json'
    temp_img_path = 'temp_img'

    try:
        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)

    except FileNotFoundError:
        print(f"{json_file_path} file not found. Correct path is data/raw/annotations.json")

        sys.exit(1)


    convert_coco_to_yolo_obj = ConvertCocoToYolo(coco_data, temp_img_path)

    convert_coco_to_yolo_obj.convert_coco_format_to_yolo()


    train_valid_data_files_obj = TrainValidDataFiles()

    train_valid_data_files_obj.split_data()