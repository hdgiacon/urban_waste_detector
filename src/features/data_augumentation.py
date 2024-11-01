import albumentations as A
import cv2

class DataAugumentation:
    @classmethod
    def __load_image_and_labels(cls, image_path, label_path):
        image = cv2.imread(image_path)
        boxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([x_center, y_center, width, height])
                class_labels.append(int(class_id))
        
        return image, boxes, class_labels
    

    @classmethod
    def __save_augmented_labels(cls, label_path, augmented_bboxes, augmented_class_labels):
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(augmented_bboxes, augmented_class_labels):
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


    @staticmethod
    def augment_image_and_labels(image_path: str, label_path: str, output_image_path: str, output_label_path: str):
        image, boxes, class_labels = DataAugumentation.__load_image_and_labels(image_path, label_path)

        try:
            # Definir o pipeline de augmentação
            augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p = 0.5),
                A.RandomRotate90(p = 0.5),
                A.ShiftScaleRotate(scale_limit = 0.1, rotate_limit = 45, shift_limit = 0.1, p = 0.5),
            ], bbox_params = A.BboxParams(format = 'yolo', label_fields = ['class_labels']))

            # Aplicar as augmentações
            augmented = augmentation_pipeline(image=image, bboxes=boxes, class_labels=class_labels)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_class_labels = augmented['class_labels']

            img_path_split = image_path.split('.')
            aug_img_name = img_path_split[0] + '_aug.' + img_path_split[1]

            # Salvar a imagem aumentada
            cv2.imwrite(aug_img_name, augmented_image)

            label_path_split = label_path.split('.')
            aug_label_name = label_path_split[0] + '_aug.' + label_path_split[1]

            # Salvar as anotações aumentadas
            DataAugumentation.__save_augmented_labels(aug_label_name, augmented_bboxes, augmented_class_labels)

        except ValueError:
            return