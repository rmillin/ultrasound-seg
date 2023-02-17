import numpy as np
import glob
import os
import cv2
import random

from tensorflow import keras
from scipy.interpolate import griddata
from deform_image import deform_frame


def deform_frame(frame, num_pts=5, std_displacement=20, mean_displacement=0):
    """
    Warp an image, using the method outlined in
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    :param frame: 2D numpy array with image
    :param num_pts: The number of points along a dimension to displace to determine warp
    :param mean_displacement: Mean of displacement for these points (random, gaussian distributed displacement)
    :param std_displacement: Standard deviation of displacement for these points
    :return: warped frame image with the same dimensions
    """

    # get the displacement values; x, y
    def_grid = np.random.randn(num_pts, num_pts, 2) * std_displacement + mean_displacement

    # get the coordinates of the frame
    x, y = np.meshgrid(np.arange(0, frame.shape[1]), np.arange(0, frame.shape[0]))

    # get the locations of the displacement values
    locations_x = np.linspace(0, frame.shape[1], num_pts)
    locations_y = np.linspace(0, frame.shape[0], num_pts)
    warped_x, warped_y = np.meshgrid(locations_x, locations_y)
    orig_x = warped_x + def_grid[:, :, 1]
    orig_y = warped_y + def_grid[:, :, 0]

    # Interpolate the warp coordinates on the image grid
    grid_z = griddata(np.concatenate((np.expand_dims(orig_x.flatten(), axis=1),
                                      np.expand_dims(orig_y.flatten(), axis=1)), axis=1),
                      np.concatenate((np.expand_dims(warped_x.flatten(), axis=1),
                                      np.expand_dims(warped_y.flatten(), axis=1)), axis=1),
                      (x, y),
                      method='cubic')
    # separate back into x and y
    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(x.shape)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(x.shape)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    # map the image to the new coordinates
    deformed_frame = cv2.remap(frame, map_y_32, map_x_32, cv2.INTER_CUBIC)

    return deformed_frame


"""
Below code adpated from DataGenerator written by Courosh Mehanian
"""


class SegmentationDataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self,
                 image_params,
                 classes,
                 labels,
                 batch_size=32,
                 shuffle=True,
                 loss='binary_ce'):
        """
        Constructor.
        """

        if image_params is not None:
            self.image_shape = image_params['image_size']
        else:
            self.image_shape = (0, 0, 0)
        self.n_classes = len(labels)
        self.batch_size = batch_size

        # generate data
        image_dir = image_params['image_dir']
        # get the list of label files, then for each find the matching image
        all_label_files = glob.glob(os.path.join(image_dir, '*_mask.tif'))
        # UNCOMMENT BELOW FOR TESTING
        # all_label_files = all_label_files[:50]
        all_image_files = [file.replace('_mask', '') for file in all_label_files]

        images = np.zeros(((len(all_image_files),) + self.image_shape[:2]), dtype=np.float32)
        image_labels = np.zeros(((len(all_image_files),) + self.image_shape[:2]), dtype=np.float32)
        for count, image_file in enumerate(all_image_files):
            print('loading image ' + str(count))
            try:
                tmp_image = cv2.imread(image_file, 0)
                tmp_label = cv2.imread(all_label_files[count], 0)
                tmp_image = cv2.resize(tmp_image, self.image_shape[:2])
                tmp_label = cv2.resize(tmp_label, self.image_shape[:2])
                tmp_label[tmp_label < 128] = 0
                tmp_label[tmp_label >= 128] = 1
            except:
                print('Skipping ' + image_file)
                continue
            images[count, :, :] = np.expand_dims(tmp_image, axis=0)
            image_labels[count, :, :] = np.expand_dims(tmp_label, axis=0)

        self.labels = labels
        self.classes = classes
        self.images = images
        self.image_labels = image_labels
        self.loss = loss
        self.n_samples = len(images)
        self.indices = np.arange(self.n_samples)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        :return: number of batches per epoch
        """
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        """
        Generates data containing a batch of images.
        X : (batch_size, n_rows, n_cols, n_channels)
        :param index: index of batch
        :return: X, y images and labels
        """
        # generate indices of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        x, y = self.__data_generation(batch_indices)

        # pass back images and labels for one batch
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)
        return

    def augment(self, image):
        # pick a random augmentation
        randint = random.choice([0, 1, 2])
        if randint == 0:
            gamma = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
            image = image.copy().astype(np.uint8)
            t = (np.power(np.linspace(0, 1, 256), 1.0 / gamma) * 255).astype(np.uint8)
            image = cv2.LUT(image, t)  # gamma adjust
        elif randint == 1:
            kernel_size = random.choice([2, 3, 4, 5])
            image = cv2.blur(image, (kernel_size, kernel_size))
        else:
            delta = 10
            pct_pix = 5
            sz = image.shape
            pix = np.random.randint(0, np.product(sz[:2]), [np.int(np.product(sz[:2]) * pct_pix * 0.01), ])
            image = image.flatten()
            image[pix] = image[pix] + delta
            image = image.reshape(sz)
            image[image < 0] = 0
            image[image > 255] = 255
        return image

    def __data_generation(self, batch_indices):

        # Prepare the sample

        # initialization
        image = np.empty((self.batch_size,) + self.image_shape, dtype=np.float32)
        y = np.empty((self.batch_size,) + self.image_shape, dtype=np.float32)

        # generate data
        for ind, idx in enumerate(batch_indices):
            # load and standardize image
            if np.sum(self.image_shape):
                tmp_image = self.images[idx, :, :]
                tmp_image = self.augment(tmp_image)
                # tmp_image = (tmp_image - np.mean(tmp_image))/np.std(tmp_image)
                tmp_image = tmp_image/255
                tmp_image = np.expand_dims(tmp_image, axis=2)
                for _ in range(self.image_shape[2] - 1):
                    tmp_image = np.concatenate((tmp_image, tmp_image), axis=2)
                if tmp_image.shape[2] > 3:
                    tmp_image = tmp_image[:, :, :3]
                image[ind] = tmp_image
                # grab label
                y[ind] = np.expand_dims(self.image_labels[idx], axis=2).astype(float)

        return image, y


class ClassificationDataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self,
                 image_params,
                 classes,
                 labels,
                 batch_size=32,
                 shuffle=True,
                 loss='binary_ce',
                 augment=True):
        """
        Constructor.
        """

        if image_params is not None:
            self.image_shape = image_params['image_size']
        else:
            self.image_shape = (0, 0, 0)
        self.n_classes = len(labels)
        self.batch_size = batch_size

        # generate data
        image_dir = image_params['image_dir']
        # get the list of label files, then for each find the matching image
        all_label_files = glob.glob(os.path.join(image_dir, '*_mask.tif'))
        # UNCOMMENT BELOW FOR TESTING
        # all_label_files = all_label_files[:50]
        all_image_files = [file.replace('_mask', '') for file in all_label_files]

        images = np.zeros(((len(all_image_files),) + self.image_shape[:2]), dtype=np.float32)
        image_labels = np.zeros((len(all_image_files), 1), dtype=np.float32)
        for count, image_file in enumerate(all_image_files):
            print('loading image ' + str(count))
            try:
                tmp_image = cv2.imread(image_file, 0)
                tmp_label = cv2.imread(all_label_files[count], 0)
                tmp_image = cv2.resize(tmp_image, self.image_shape[:2])
                if np.sum(tmp_label) > 0:
                    tmp_label = 1
                else:
                    tmp_label = 0
            except:
                print('Skipping ' + image_file)
                continue
            images[count, :, :] = np.expand_dims(tmp_image, axis=0)
            image_labels[count] = tmp_label

        self.image_labels = image_labels
        self.image_shape = image_params['image_size']
        self.positive_indices = [ind for ind, el in enumerate(self.image_labels) if el]
        self.negative_indices = [ind for ind, el in enumerate(self.image_labels) if not el]
        self.classes = classes
        self.images = images
        self.loss = loss
        self.n_samples = len(images)
        self.indices = np.arange(self.n_samples)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment = augment

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        :return: number of batches per epoch
        """
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        """
        Generates data containing a batch of images.
        X : (batch_size, n_rows, n_cols, n_channels)
        :param index: index of batch
        :return: X, y images and labels
        """
        # generate indices of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = [self.image_labels[k] for k in batch_indices]

        x, y = self.__data_generation(batch_indices, batch_labels)

        # pass back images and labels for one batch
        return x, y

    def augment_data(self, image):
        # pick a random augmentation
        aug = [random.choice([True, False]), random.choice([True, False]), random.choice([True, False]),
               random.choice([True, False]), random.choice([True, False]), random.choice([True, False]), random.choice([True, False])]
        aug_types = list(range(len(aug)))
        # random.shuffle(aug_types)
        for ind, aug_type in enumerate(aug_types):
            if aug[ind]:
                if aug_type == 0:
                    image = np.fliplr(image)  # flip left right
                elif aug_type == 1:
                    gamma = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
                    t = np.power(np.linspace(0, 1, 256), gamma)
                    # put through gamma lookup table, output is float32 between [0, 1]
                    image = cv2.LUT(image.astype(np.int8), t)
                    if image.max() <= 1:  # to handle different versions of opencv
                        image = image * 255
                elif aug_type == 2:
                    kernel_size = random.choice([2, 3, 4, 5])
                    image = cv2.blur(image, (kernel_size, kernel_size))
                elif aug_type == 3:
                    delta = 10
                    pct_pix = 5
                    sz = image.shape
                    pix = np.random.randint(0, np.product(sz[:2]), [np.int(np.product(sz[:2]) * pct_pix * 0.01), ])
                    image = image.flatten()
                    image[pix] = image[pix] + delta
                    image = image.reshape(sz)
                    image[image < 0] = 0
                    image[image > 255] = 255
                elif aug_type == 4:
                    # shift image
                    horiz_shift = random.choice([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
                    vert_shift = random.choice([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10])
                    if horiz_shift > 0:
                        image = image[:, :-horiz_shift]
                        image = np.append(np.zeros((image.shape[0], horiz_shift)), image, axis=1)
                    else:
                        image = image[:, -horiz_shift:]
                        image = np.append(image, np.zeros((image.shape[0], -horiz_shift)), axis=1)
                    if vert_shift > 0:
                        image = image[:-vert_shift, :]
                        image = np.append(np.zeros((vert_shift, image.shape[1])), image, axis=0)
                    else:
                        image = image[-vert_shift:, :]
                        image = np.append(image, np.zeros((-vert_shift, image.shape[1])), axis=0)
                elif aug_type == 5:
                    # shift image
                    old_shape = image.shape
                    expand_contract = random.choice([.9, .95, 1.05, 1.1])
                    new_shape = [int(old_shape[0] * expand_contract), int(old_shape[1] * expand_contract)]
                    image = cv2.resize(image, (new_shape[0], new_shape[1]))
                    shift_x1 = int(np.floor((new_shape[1] - old_shape[1]) / 2))
                    shift_x2 = int(np.ceil((new_shape[1] - old_shape[1]) / 2))
                    shift_y1 = int(np.floor((new_shape[0] - old_shape[0]) / 2))
                    shift_y2 = int(np.ceil((new_shape[0] - old_shape[0]) / 2))
                    if expand_contract < 1:
                        image = np.append(np.append(np.zeros((new_shape[0], -shift_x1)), image, axis=1), np.zeros((new_shape[0], -shift_x2)), axis=1)
                        image = np.append(np.append(np.zeros((-shift_y1, old_shape[1])), image, axis=0), np.zeros((-shift_y2, old_shape[1])), axis=0)
                    else:
                        image = image[shift_x1:-shift_x2, shift_y1:-shift_y2]
                else:
                    pix_std = round(image.shape[0] / 100)
                    image = deform_frame(image, std_displacement=pix_std)
            else:
                pass

        return image

    def on_epoch_end(self):
        """
        Updates indices after each epoch.
        """
        # to create balance, deal separately with positive and negative samples
        # if there are fewer of one class, append a random sampling of indices
        # then interweave positive and negative indices (after shuffling each independently if requested)
        # so batches will be balanced
        # deal with positive indices
        positive_indices = self.positive_indices.copy()
        negative_indices = self.negative_indices.copy()
        if len(positive_indices) > len(negative_indices):
            negative_indices.extend(np.random.choice(negative_indices, (len(positive_indices) - len(negative_indices),)).tolist())
        elif len(negative_indices) > len(positive_indices):
            positive_indices.extend(np.random.choice(positive_indices, (len(negative_indices) - len(positive_indices),)).tolist())

        # if self.shuffle:
        #     np.random.shuffle(positive_indices)
        #     np.random.shuffle(negative_indices)

        indices = []
        for ind in range(len(positive_indices)):
            indices.extend([positive_indices[ind], negative_indices[ind]])
        self.indices = np.array(indices)
        return

    def __data_generation(self, batch_indices, batch_labels):

        # Prepare the sample

        # initialization
        X = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
        y = np.empty((self.batch_size,), dtype=np.float32)

        # generate data
        pairs = zip(batch_indices, batch_labels)

        # read each image in batch
        for i, pair in enumerate(pairs):
            img = self.images[pair[0]].copy()
            if self.augment:
                img = self.augment_data(img)
            img = img / 255  # scale to 0-1
            # cv2.imwrite(os.path.join(r"/con_code/adult_con", "image" + str(random.randint(0, 1000)) + ".png"), img * 255)
            img = np.expand_dims(img, axis=2)
            X[i,] = np.expand_dims(img, axis=0)
            y[i] = pair[1]
            # with open(r'/home/rmillin/Documents/tmp/C/batched_indices_' + str(random.randint(1, 10000)) + '.txt', "a") as myfile:
            #     myfile.write(str(pair[0]) + '\n')

        # convert y to one-hot form
        z = keras.utils.to_categorical(y, num_classes=self.n_classes).astype(np.float32)

        return X, z

