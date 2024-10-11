import os
import json

from .convert_coco_to_yolo import ConvertCocoToYolo
from .train_valid_data_files import TrainValidDataFiles

# TODO: trocar map por numpy

def main_features() -> None:
    '''
    Call all necessary methods to create desired features for training.

    Return:
        No data. Save all features on `data/processed/` directory.
    '''

    if not os.path.isdir('data/processed/images') or not os.path.isdir('data/processed/labels'):
        json_file_path = 'data/raw/annotations.json'
        temp_img_path = 'temp_img'

        with open(json_file_path, 'r') as f:
            coco_data = json.load(f)


        convert_coco_to_yolo_obj = ConvertCocoToYolo(coco_data, temp_img_path)

        convert_coco_to_yolo_obj.convert_coco_format_to_yolo()


        train_valid_data_files_obj = TrainValidDataFiles()

        train_valid_data_files_obj.split_data()