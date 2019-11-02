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
                 classes,
                 labels,
                 batch_size=32,
                 shuffle=True,
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
        # get the list of label files, then for each find the matching image
        all_label_files = glob.glob(os.path.join(image_dir, '*_mask.tif'))
        # UNCOMMENT BELOW FOR TESTING
        all_label_files = all_label_files[:50]
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
        self.custom_loss = custom_loss
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
        y = np.empty((self.batch_size,) + self.image_shape, dtype=np.int32)

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
                y[ind] = np.expand_dims(self.image_labels[idx], axis=2)

        return image, y
