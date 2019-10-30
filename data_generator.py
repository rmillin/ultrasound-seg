import keras
import numpy as np
import glob
import os
import cv2
import random


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self,
                 image_params,
                 labels,  # labels need to match names of subfolders with images from that class
                 batch_size=32,
                 shuffle=True,
                 mask=None,
                 custom_loss=False):
        """
        Constructor.
        If augmenting, batch size will be the video batch size; number of m-/d-mode samples will be multiplied.
        If starting from video, video_params must be provided
        Otherwise, if using fusion model, image_params and d_mode_params must be provided; otherwise only the corresponding params
        """

        if image_params is not None:
            self.image_shape = image_params['image_size']
        else:
            self.image_shape = (0, 0, 0)
        self.n_classes = len(labels)
        self.batch_size = batch_size

        # generate data
        image_dir = image_params['image_dir']
        all_files = []
        class_labels = np.empty((0, 1))
        for ind, label in enumerate(labels):
            # load and standardize image
            image_files = glob.glob(os.path.join(image_dir, label, '*.jpg'))
            # uncomment below for testing with small batch
            # image_files = image_files[:100]
            all_files.extend(image_files)
            labels = np.ones((len(image_files), 1)) * ind
            class_labels = np.concatenate((class_labels, labels), axis=0)

        images = np.zeros(((len(all_files),) + self.image_shape[:2]), dtype=np.float32)
        for count, image_file in enumerate(all_files):
            if np.sum(self.image_shape):
                print('loading image ' + str(count))
                tmp_image = cv2.imread(image_file, 0)
                # tmp_image = (tmp_image - np.mean(tmp_image))/np.std(tmp_image)
                tmp_image = cv2.resize(tmp_image, self.image_shape[:2])
                # change to inverted image
                # tmp_image = 255 - tmp_image
                images[count, :, :] = np.expand_dims(tmp_image, axis=0)

        self.labels = class_labels
        self.images = images
        self.mask = mask
        self.custom_loss = custom_loss
        self.n_samples = len(class_labels)
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
        randint = random.choice([0, 1, 2, 3])
        if randint == 0:
            image = np.fliplr(image)  # flip left right
        elif randint == 1:
            gamma = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
            image = image.copy().astype(np.uint8)
            t = (np.power(np.linspace(0, 1, 256), 1.0 / gamma) * 255).astype(np.uint8)
            image = cv2.LUT(image, t)  # gamma adjust
        elif randint == 2:
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
        y = np.empty((self.batch_size,) + (1,), dtype=np.int32)

        # generate data
        for ind, idx in enumerate(batch_indices):
            # load and standardize image
            if np.sum(self.image_shape):
                tmp_image = self.images[idx, :, :]
                tmp_image = self.augment(tmp_image)
                tmp_image = (tmp_image - np.mean(tmp_image))/np.std(tmp_image)
                tmp_image = np.expand_dims(tmp_image, axis=2)
                for _ in range(self.image_shape[2] - 1):
                    tmp_image = np.concatenate((tmp_image, tmp_image), axis=2)
                if tmp_image.shape[2] > 3:
                    tmp_image = tmp_image[:, :, :3]
                image[ind] = tmp_image
                # grab label
                y[ind] = self.labels[idx]

        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        if (self.mask is not None) and not self.custom_loss:
            return [image, self.mask], y

        return image, y