import os
import shutil
import copy

from .data_augumentation import DataAugumentation

class TrainValidDataFiles:
    '''
    Class for dividing the data of all classes respecting their correct proportionality, so that there is no imbalance in the training stage and thus 
    incorrect classifications.
    '''

    def __init__(self) -> object:
        TrainValidDataFiles.__images_train_path = 'data/processed/images/train/'
        TrainValidDataFiles.__images_valid_path = 'data/processed/images/val/'
        TrainValidDataFiles.__labels_train_path = 'data/processed/labels/train/'
        TrainValidDataFiles.__labels_valid_path = 'data/processed/labels/val/'

    @classmethod
    def __create_folders_for_yolo(cls) -> None:
        '''
        Private method to create necessary folders for YOLO training if they not exist.          

        Return:
            No data. Create folders with `os.makedirs`.
        '''

        os.makedirs(cls.__images_train_path, exist_ok = True)
        os.makedirs(cls.__images_valid_path, exist_ok = True)
        os.makedirs(cls.__labels_train_path, exist_ok = True)
        os.makedirs(cls.__labels_valid_path, exist_ok = True)


    @classmethod
    def __divide_elements(_, num_elements) -> int | int:
        '''
        Private method to calculate the number of elements of the training and validation bases by the number of data.          

        Args:
            num_elements: total number of elements in a given database.

        Return:
            Number of elements of the `training` base, number of elements of the `validation` base.
        '''

        if num_elements <= 2:
            return num_elements, 0

        eighty_percent = round(num_elements * 0.8)
        twenty_percent = round(num_elements * 0.2)

        if eighty_percent + twenty_percent != num_elements:
            eighty_percent = num_elements - twenty_percent

        return eighty_percent, twenty_percent


    @classmethod
    def __train_valid_split_data(_, all_current_data_class_path: list, train_size: int) -> list[str] | list[str]:
        '''
        Private method to separate the elements of the database into training and testing by means of the correct amount. A correction is made if the 
        image file is not together with its corresponding .txt.          

        Args:
            all_current_data_class_path: all data paths of a given class of data objects;
            train_size: number of elements that the database must have.

        Return:
            `Training` base path list, `validation` base path list.
        '''

        train_list = copy.deepcopy(sorted(all_current_data_class_path)[:train_size])
        valid_list = copy.deepcopy(sorted(all_current_data_class_path)[train_size:])

        last_element_train_list = train_list[-1].split('.')[0]
        first_element_valid_list = valid_list[0].split('.')[0] if valid_list else None

        if last_element_train_list == first_element_valid_list:
            train_size += 1

            train_list = copy.deepcopy(sorted(all_current_data_class_path)[:train_size])
            valid_list = copy.deepcopy(sorted(all_current_data_class_path)[train_size:])

        return train_list, valid_list


    @classmethod
    def __move_one_img_or_txt_file(_, file_name: str, current_temp_class_folder_path: str, images_path: str, labels_path: str) -> None:
        '''
        Private method to move the current image or text file to its correct folder in `data/processed`.          

        Args:
            file_name: current file name with it's extension;
            current_temp_class_folder_path: current file folder full path;
            images_path: images path where current file is going to be sent;
            labels_path: labels path where current file is going to be sent.

        Return:
            No data. `shutil.mode` will move the files.
        '''

        _, file_extension = os.path.splitext(file_name)
        
        if file_extension.lower() == '.txt':
            shutil.move(os.path.join(current_temp_class_folder_path, file_name), labels_path)

        else:
            shutil.move(os.path.join(current_temp_class_folder_path, file_name), images_path)


    @classmethod
    def __move_img_and_txt_files(cls, cathegory_folder: str, directory: str) -> None:
        '''
        Private method for moving all image files and .txt of a class to their respective folders in `data/processed`.

        Args:
            cathegory_folder: current cathegory folder;
            directory: all cathegorie folders directory path.

        Return:
            No data.
        '''

        current_temp_class_folder_path = os.path.join(directory, cathegory_folder)
        
        num_elements_not_aug = len(copy.deepcopy(os.listdir(current_temp_class_folder_path)))

        if num_elements_not_aug / 2 >= 10:
            all_data_not_aug = copy.deepcopy(sorted(os.listdir(current_temp_class_folder_path)))
            
            if num_elements_not_aug / 2 < 50:
                for index, element in enumerate(all_data_not_aug):
                    if element.endswith('.txt'):
                        continue

                    DataAugumentation.augment_image_and_labels(
                        os.path.join(current_temp_class_folder_path, element),
                        os.path.join(current_temp_class_folder_path, all_data_not_aug[index + 1]),
                        current_temp_class_folder_path,
                        current_temp_class_folder_path
                    )

            #with open('meu_arquivo.txt', 'a') as arquivo:
            #    arquivo.write(f"Class folder: {cathegory_folder} -> num of elements: {num_elementos}\n")

            all_data_with_aug = os.listdir(current_temp_class_folder_path)

            train_size, valid_size = cls.__divide_elements(len(all_data_with_aug))

            train_list, valid_list = cls.__train_valid_split_data(all_data_with_aug, train_size)

            list(map(lambda file_name: cls.__move_one_img_or_txt_file(
                file_name, 
                current_temp_class_folder_path, 
                cls.__images_train_path, 
                cls.__labels_train_path), train_list
            ))

            list(map(lambda file_name: cls.__move_one_img_or_txt_file(
                file_name, 
                current_temp_class_folder_path, 
                cls.__images_valid_path, 
                cls.__labels_valid_path), valid_list
            ))


    @staticmethod
    def split_data() -> None:
        '''
        Static method for dividing all images and files .txt of all classes according to their correct proportion. The idea is to avoid imbalance between the classes.          

        Return:
            No data. Files will be moved.
        '''

        directory = 'temp_img/'

        TrainValidDataFiles.__create_folders_for_yolo()

        list(map(lambda cathegory_folder: TrainValidDataFiles.__move_img_and_txt_files(cathegory_folder, directory), os.listdir(directory)))

        shutil.rmtree(directory)

        print("Divisão concluída.")