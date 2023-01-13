import load_custom_model as lcm
from tensorflow.keras.models import load_model
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

    # with open(model_path, 'rb') as f:
    #     binary_model = pickle.load(f)
    # binary_model = tf.saved_model.load(model_path)

    test_images = glob.glob(os.path.join(test_data_path, '*.tif'))
    test_images = [image for image in test_images if 'mask' not in image]
    # UNCOMMENT FOR TESTING ONLY
    test_images = test_images[:50]
    gt_classes = []
    pred_classes = []
    dice_coeffs = []
    for image_file in test_images:
        mask_file = glob.glob(image_file.replace(test_data_path, test_label_path).replace('.tif', '_mask.tif'))
        if not len(mask_file):
            print('No label available for ' + image_file + ', skipping.')
            continue
        mask_file = mask_file[0]
        image = cv2.imread(image_file)
        image_size = image.shape
        mask = cv2.imread(mask_file)
        image = cv2.resize(image, tuple(input_size[1:3]))[:, :, 0]
        mask = cv2.resize(mask, tuple(input_size[1:3]))[:, :, 0]
        # image = cv2.resize(image, binary_model.image_params['image_size'][:2])[:, :, 0]
        # mask = cv2.resize(mask, binary_model.image_params['image_size'][:2])[:, :, 0]
        image = image.astype(np.float32) / 255
        image = np.expand_dims(np.expand_dims(image, axis=2), axis=0)
        gt = int(np.sum(mask) > 0)
        gt_classes.append(gt)

        if classify:
            classification_model = load_model(classification_model_path)
            pred_class = classification_model(image).numpy()
            print(pred_class)
            pred_class = np.round(pred_class[0][1].squeeze())

        # pred = binary_model.predict(image)
        if segment:
            segmentation_model = lcm.load_custom_model(segmentation_model_path, segmentation_loss, size=input_size)
            pred = segmentation_model(image).numpy()
            pred = pred[0].squeeze()
            pred = pred > 0.5
            dice = calc_dice(mask, pred)
            print(dice)
            dice_coeffs.append(dice)
            pred = cv2.resize(pred.astype(float) * 255, (image_size[1], image_size[0]))
            cv2.imwrite(mask_file.replace('.tif', '_pred.tif'), pred.astype(int))

        # output results based on requested process (classification only, segmentation only, or both)
        if classify:
            if segment:
                if pred_class and np.sum(pred):
                    pred_classes.append(1)

                else:
                    pred_classes.append(0)
            else:
                pred_classes.append(pred_class)
        else:
            pred_classes.append(int(np.sum(pred) > 0))


    return


if __name__ == '__main__':
    # model_path = '/Users/rmillin/Downloads/ultrasound-nerve-segmentation/results/debug/binary_model_instance.pickle'
    segmentation_model_path = '/Users/rmillin/Downloads/ultrasound-nerve-segmentation/results/debug/dice/trained_fusion_model.h5'
    classification_model_path = '/Users/rmillin/Downloads/ultrasound-nerve-segmentation/results/debug/classification/trained_fusion_model.h5'
    segmentation_loss = 'dice'
    test_data_path = '/Users/rmillin/Downloads/ultrasound-nerve-segmentation/train'
    test_label_path = '/Users/rmillin/Downloads/ultrasound-nerve-segmentation/train'
    classify = False
    segment = True
    input_size = [4, 128, 128, 1]
    main()
