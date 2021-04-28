import os
import shutil
from glob import glob


def copy_all_file_from_folder_to_folder():
    list_folder = glob("/home/vuong/Desktop/Data/*")
    path_train_save = "dataset3/train/"
    path_label_save = "dataset3/label/"
    if not os.path.exists(path_train_save):
        os.makedirs(path_train_save)

    if not os.path.exists(path_label_save):
        os.makedirs(path_label_save)

    for folder_cp in list_folder:
        path_train = folder_cp + "/default/*"
        path_label = folder_cp + "/defaultannot/*"
        list_path_image_train = glob(path_train + '*')
        list_path_image_label = glob(path_label + '*')
        for file in list_path_image_train:
            shutil.copy2(file, os.path.join(path_train_save, file.split('/')[-1]))
        for file in list_path_image_label:
            shutil.copy2(file, os.path.join(path_label_save, file.split('/')[-1]))


if __name__ == '__main__':
    copy_all_file_from_folder_to_folder()
