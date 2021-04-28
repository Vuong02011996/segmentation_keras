import cv2
import numpy as np
from glob import glob
import os


def convert_file_label_to_camvid_data_training():
    label_file = "label_colors.txt"
    with open(label_file, "r+") as fr:
        lines = fr.readlines()
        rgb_list = []
        label_list = []
        for line in lines:
            line = line.rstrip()
            rgb_list.append([int(line.split(' ')[2]), int(line.split(' ')[1]), int(line.split(' ')[0])])
            label_list.append(' '.join(line.split(' ')[3:]))
    return rgb_list


def check_label_after_convert():
    path = "test.png"
    image = cv2.imread(path)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            print(image[row, col])


def convert_label_using_numpy_array():
    path = "dataset3/label/"
    path_save = "dataset3/label_convert/"
    rgb_list = convert_file_label_to_camvid_data_training()
    list_image = glob(path + "*")
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx, image_path in enumerate(list_image):
        print("{}/{}".format(idx, len(list_image)), "processing image: ", image_path)
        image = cv2.imread(image_path)
        for label in rgb_list:
            index_row, index_col = np.where((image[:, :, 0] == label[0]) & (image[:, :, 1] == label[1]) & (image[:, :, 2] == label[2]))
            for i in range(len(index_col)):
                image[index_row[i], index_col[i]] = rgb_list.index(label)
        cv2.imwrite(path_save + image_path.split("/")[-1], image)


def main():
    path = "dataset2/annotations_prepped_train/"
    path_save = "dataset2/annotations_prepped_train_convert/"
    rgb_list = convert_file_label_to_camvid_data_training()

    list_image = glob(path + "*")
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    for idx, image_path in enumerate(list_image):
        print("{}/{}".format(idx, len(list_image)), "processing image: ", image_path)
        image = cv2.imread(image_path)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                for label in rgb_list:
                    if label == list(image[row, col]):
                        image[row, col] = rgb_list.index(label)
                        break
        cv2.imwrite(path_save + image_path.split("/")[-1], image)


if __name__ == '__main__':
    # convert_file_label_to_camvid_data_training()
    # main()
    convert_label_using_numpy_array()