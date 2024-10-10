import os
import shutil

class ConvertCocoToYolo:
    ''''''

    def __init__(
        self, 
        coco_data: dict, 
        temp_img_path: str
    ) -> object:
        ConvertCocoToYolo.__coco_data = coco_data
        ConvertCocoToYolo.__temp_img_path = temp_img_path
        
    
    @classmethod
    def __adjust_and_move_img(
        _, 
        annotation: dict, 
        img_file_name: str, 
        cathegorie_folder_path: str, 
        img_file_name_split: list, 
    ) -> None:
        '''
        Private method to adjust the image name so that it is unique, as there is data with the same name. It also moves the image file to a temporary 
        folder `temp_img/`.          

        Args:
            annotation: dictionary containing the current annotation of the data, such as data from the bounding boxes and which image it refers to;
            img_file_name: image file name;
            cathegorie_folder_path: folder path of the current data class;
            img_file_name_split: Image name tokenized by the character `/`.

        Return:
            No data. `shutil.copy` send a copy of current image to it's correct destiny.
        '''

        img_file_name_split_2 = img_file_name_split[1].split('.')

        img_origin_path = os.path.join('data/raw', img_file_name)
        img_destiny_path = os.path.join(cathegorie_folder_path, img_file_name_split[0] + '_' + img_file_name_split_2[0] + '_' + str(annotation['category_id']) + '.' + img_file_name_split_2[1])

        if not os.path.isfile(img_destiny_path):
            shutil.copy(img_origin_path, img_destiny_path)


    @classmethod
    def __coco_bbox_to_yolo(_, bbox: list, img_width: int, img_height: int) -> list[float]:
        '''
        Private method for converting bbox from the `COCO` format (xmin, ymin, width, height) to `YOLO` (x_center, y_center, width, height)          

        Args:
            bbox: list with the bounding box data in COCO;
            img_width: image width;
            img_height: image height;

        Return:
            A list with YOLO boundding box data.
        '''

        x_min, y_min, box_width, box_height = bbox
        x_center = x_min + (box_width / 2)
        y_center = y_min + (box_height / 2)
        
        
        ## Normalize coordinates (dividing by the width and height of the image)
        
        x_center /= img_width
        y_center /= img_height
        box_width /= img_width
        box_height /= img_height
        
        return [x_center, y_center, box_width, box_height]
    
    
    @classmethod
    def __get_yolo_bbox(cls, annotation: dict) -> list[float]:
        '''
        private method that, from the data in the `COCO` format, creates the bounding box for the `YOLO` format.          

        Args:
            annotation: dictionary containing the current annotation of the data, such as data from the bounding boxes and which image it refers to.

        Return:
            A list with YOLO boundding box data.
        '''

        image_id = annotation['image_id']
        bbox = annotation['bbox']
        
        image_info = next(img for img in cls.__coco_data['images'] if img['id'] == image_id)
        img_width = image_info['width']
        img_height = image_info['height']
        
        return cls.__coco_bbox_to_yolo(bbox, img_width, img_height)
    

    @classmethod
    def __create_txt_file(
        _, 
        annotation: dict, 
        yolo_bbox: list, 
        cathegorie_folder_path: str, 
        img_file_name_split: list
    ) -> None:
        '''
        Private method that creates .txt file from bounding box converted data.          

        Args:
            _: _description_
            annotation: dictionary containing the current annotation of the data, such as data from the bounding boxes and which image it refers to;
            yolo_bbox: list with YOLO boundding box data;
            cathegorie_folder_path: current class folder path;
            img_file_name_split: Image name tokenized by the character `/`.

        Return:
            No data. TextIOWrapper`.write` create and saves `.txt` file.
        '''

        txt_path = os.path.join(cathegorie_folder_path, img_file_name_split[0] + '_' + img_file_name_split[1].split('.')[0] + '_' + str(annotation['category_id']) + '.txt')

        with open(txt_path, 'a+') as f_txt:
            f_txt.seek(0)
            content = f_txt.read()

            if content:
                f_txt.write('\n')

            # YOLO format: <category_id> <x_center> <y_center> <width> <height>
            f_txt.write(f"{annotation['category_id']} {' '.join(map(str, yolo_bbox))}")


    @classmethod
    def __convert_one_data(cls, annotation: dict) -> None:
        '''
        Private method that convert one COCO data to YOLO format.          

        Args:
            annotation: dictionary containing the current annotation of the data, such as data from the bounding boxes and which image it refers to;

        Return:
            No data.
        '''

        cathegorie_folder_path = os.path.join(cls.__temp_img_path, str(annotation['category_id']))
        os.makedirs(cathegorie_folder_path, exist_ok = True)

        img_file_name = cls.__coco_data['images'][annotation['image_id']]['file_name']
        img_file_name_split = img_file_name.split('/')

        cls.__adjust_and_move_img(annotation, img_file_name, cathegorie_folder_path, img_file_name_split)

        yolo_bbox = cls.__get_yolo_bbox(annotation)

        cls.__create_txt_file(annotation, yolo_bbox, cathegorie_folder_path, img_file_name_split)


    @staticmethod
    def convert_coco_format_to_yolo() -> None:
        '''Static method that convert all `COCO` format data present on `data/raw/anootations.json` to `YOLO` format as .txt files.      

        Return:
            No data. Files will be created and moved.
        '''

        list(map(lambda annotantion: ConvertCocoToYolo.__convert_one_data(annotantion), ConvertCocoToYolo.__coco_data['annotations']))

        print("Conversão concluída.")