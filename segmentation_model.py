"""
Class defining the fusion model architecture
Option to build fusion, m-mode only, or d-mode only
"""

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras import optimizers, regularizers
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import glob
import os

from data_generator import DataGenerator


class BinaryModel:
    # class defining the multilabel model.
    # functions for defining the model, training, and prediction
    # model type can be resnet, inception, or mobilenet

    def __init__(self, image_params, model_type='basic', classes=('background', 'foreground'), labels=(0, 1),
                 custom_loss=False, batch_size=4):
        self.classes = classes
        self.labels = labels
        self.batch_size = batch_size

        # Define custom loss
        # Define custom loss
        def weighted_crossentropy(size):
            """
            cross-entropy loss with positive class up-weighted
            of the form L = L * w if y == 1
            :param y: true label
            :param y_pred: predicted label
            :param mask: position-based weights for penalty
            :return: loss
            """

            def weighted_loss(y_true, y_pred):
                w = K.constant(np.ones(size))
                w = K.maximum(w, y_true * 98)  # make weights for positive entries 98
                w = w/K.sum(w)
                # element-wise cross-entropy
                e_ce = y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred)
                loss = K.sum(e_ce * w)
                return loss

            return weighted_loss

        if image_params is None:
            image_params = {'image_size': None}

        self.image_params = image_params
        self.custom_loss=custom_loss
        # first convolution layer

        if model_type == 'basic':
            # use the U-net stolen from: https://github.com/zhixuhao/unet

            image_input = Input(shape=self.image_params['image_size'])
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(image_input)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(drop5))
            merge6 = concatenate([drop4, up6], axis=3)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv6))
            merge7 = concatenate([conv3, up7], axis=3)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv7))
            merge8 = concatenate([conv2, up8], axis=3)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv8))
            merge9 = concatenate([conv1, up9], axis=3)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

            model = Model(input=image_input, output=conv10)
        else:
            raise('Only basic model is currently available.')

        # compile
        opt = optimizers.Adam(lr=0.0001)

        self.model = model
        if custom_loss:
            self.model.compile(optimizer=opt,
                               loss=weighted_crossentropy([batch_size,
                                                           image_params['image_size'][0],
                                                           image_params['image_size'][1], 1]),
                               metrics=['accuracy'])
        else:
            self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


    # training
    def train(self, train_image_dir, validation_image_dir=None, n_epochs=10, batch_size=32, save_dir=None):

        mc = ModelCheckpoint(filepath=os.path.join(save_dir, 'model.{epoch:02d}.hdf5'), monitor='val_loss',
                             verbose=1, period=5)
        tb = TensorBoard(log_dir=save_dir)
        # es = EarlyStopping(patience=2, monitor='val_acc', mode='max', baseline=0.93)

        # get the number of training and val examples
        n_train = len(glob.glob(os.path.join(validation_image_dir, '*_mask.tif')))
        n_val = len(glob.glob(os.path.join(validation_image_dir, '*_mask.tif')))

        image_params = self.image_params

        image_params['image_dir'] = train_image_dir

        train_image_generator = DataGenerator(image_params,
                                              self.classes,  # labels need to match names of subfolders with images from that class
                                              self.labels,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              custom_loss=self.custom_loss)

        image_params['image_dir'] = validation_image_dir

        validation_image_generator = DataGenerator(image_params,
                                                   self.classes,  # labels need to match names of subfolders with images from that class
                                                   self.labels,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   custom_loss=self.custom_loss)

        # Fit the model on the batches generated by datagen.flow().
        self.model.fit_generator(train_image_generator,
                                 steps_per_epoch=int(np.ceil(n_train / float(batch_size))),
                                 epochs=n_epochs,
                                 validation_data=validation_image_generator,
                                 validation_steps=int(np.ceil(n_val / float(batch_size))),
                                 workers=12,
                                 callbacks=[mc, tb])

        self.model.save(os.path.join(save_dir, 'trained_fusion_model.h5'))

    # prediction
    def predict(self, test_image_image=None):

        predictions = self.model.predict(test_image_image)
        classes = self.classes

        return predictions, classes
