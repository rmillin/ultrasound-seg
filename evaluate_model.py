import segmentation_model
from segmentation_model import BinaryModel
import pickle
import os
import glob
import cv2
import numpy as np


def calc_iou(y_actual, y_pred):
    # calculate intersection over union
    if (not y_actual.sum()) and (not(y_pred.sum())):
        return 1
    intersection = (y_actual & y_pred).sum()
    union = (y_actual | y_pred).sum() - intersection
    return intersection/union


def calc_dice(y_actual, y_pred):
    # calculate dice coefficient
    if (not y_actual.sum()) and (not(y_pred.sum())):
        return 1
    intersection = (y_actual & y_pred).sum()
    dice = 2 * intersection / (y_actual.sum() + y_pred.sum())
    return dice


def main():

    with open(model_path, 'rb') as f:
        binary_model = pickle.load(f)

    test_images = glob.glob(os.path.join(test_data_path, '*.tif'))
    for image_file in test_images:
        mask_file = glob.glob(image_file.replace(test_data_path, test_label_path)).replace('.tif', '_mask.tif')
        if not len(mask_file):
            print('No label available for ' + image_file + ', skipping.')
            continue
        mask_file = mask_file[0]
        image = cv2.imread(image_file)
        mask = cv2.imread(mask_file)
        image = cv2.resize(image, binary_model.image_shape[:2])
        mask = cv2.resize(mask, binary_model.image_shape[:2])
        image = (image - np.mean(image)) / np.std(image)
        image = np.expand_dims(image, axis=2)

        pred = binary_model.predict(image)
        dice = calc_dice(mask, pred > 0.5)
        print(dice)

    return


if __name__ == '__main__':
    model_path = '/Users/rmillin/Downloads/binary_model_instance_old.pickle'
    test_data_path = '/Users/rmillin/Documents/ultrasound-nerve-segmentation/val_images'
    test_label_path = '/Users/rmillin/Documents/ultrasound-nerve-segmentation/val_masks'
    main()