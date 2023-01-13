"""
Class defining the model architecture
"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, \
    concatenate, GlobalMaxPooling2D, Dense, Reshape, Flatten, Concatenate, BatchNormalization, Conv2DTranspose, LeakyReLU
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.applications import EfficientNetB2, MobileNetV2

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import glob
import os
import custom_losses as cls

from data_generator import SegmentationDataGenerator, ClassificationDataGenerator


class SegmentationModel:
    # class defining the multilabel model.
    # functions for defining the model, training, and prediction
    # model type can be resnet, inception, or mobilenet

    def __init__(self, image_params, model_type='basic', classes=('background', 'foreground'), labels=(0, 1),
                 loss='weighted_ce', batch_size=4):
        self.classes = classes
        self.labels = labels
        self.batch_size = batch_size

        if image_params is None:
            image_params = {'image_size': None}

        self.image_params = image_params
        self.loss=loss
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

            model = Model(inputs=image_input, outputs=conv10)
        elif model_type == 'resnet':
            # resnet type encoder
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
            model = Model(inputs=image_input, outputs=conv10)
        else:
            raise('Only basic model is currently available.')

        # compile
        opt = optimizers.Adam(lr=0.00001)

        self.model = model
        if loss == 'weighted_ce':
            self.model.compile(optimizer=opt,
                               loss=cls.weighted_crossentropy([batch_size, image_params['image_size'][0], image_params['image_size'][1], 1]),
                               metrics=['accuracy'])
        elif loss == 'dice':
            self.model.compile(optimizer=opt,
                               loss=cls.dice_loss(),
                               metrics=['accuracy'])
        else:
            print('unrecognized loss, using regular binary cross entropy')
            self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


    # training
    def train(self, train_image_dir, validation_image_dir=None, n_epochs=10, batch_size=32, save_dir=None):

        mc = ModelCheckpoint(filepath=os.path.join(save_dir, 'model.{epoch:02d}'), monitor='val_loss',
                             verbose=1, period=5)
        tb = TensorBoard(log_dir=save_dir)
        # es = EarlyStopping(patience=2, monitor='val_acc', mode='max', baseline=0.93)

        # get the number of training and val examples
        n_train = len(glob.glob(os.path.join(validation_image_dir, '*_mask.tif')))
        n_val = len(glob.glob(os.path.join(validation_image_dir, '*_mask.tif')))

        image_params = self.image_params

        image_params['image_dir'] = train_image_dir

        train_image_generator = SegmentationDataGenerator(image_params,
                                              self.classes,  # labels need to match names of subfolders with images from that class
                                              self.labels,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              loss=self.loss)

        image_params['image_dir'] = validation_image_dir

        validation_image_generator = SegmentationDataGenerator(image_params,
                                                   self.classes,  # labels need to match names of subfolders with images from that class
                                                   self.labels,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   loss=self.loss)

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


class ClassificationModel:
    # class defining the multilabel model.
    # functions for defining the model, training, and prediction

    def __init__(self, image_params, model_type='basic', classes=('no nerve', 'nerve'), labels=(0, 1),
                 loss='weighted_ce', batch_size=4):
        self.classes = classes
        self.labels = labels
        self.batch_size = batch_size

        if image_params is None:
            image_params = {'image_size': None}

        self.image_params = image_params
        self.loss=loss
        # first convolution layer

        if model_type == 'basic':

            image_input = Input(shape=self.image_params['image_size'])
            conv1 = Conv2D(16, 3, activation='relu')(image_input)
            conv1 = Conv2D(16, 3, activation='relu')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            norm1 = BatchNormalization()(pool1)
            conv2 = Conv2D(32, 3, activation='relu')(norm1)
            conv2 = Conv2D(32, 3, activation='relu')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            norm2 = BatchNormalization()(pool2)
            conv3 = Conv2D(64, 3, activation='relu')(norm2)
            conv3 = Conv2D(64, 3, activation='relu')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            norm3 = BatchNormalization()(pool3)
            conv4 = Conv2D(256, 3, activation='relu')(norm3)
            conv4 = Conv2D(256, 3, activation='relu')(conv4)
            drop4 = Dropout(0.5)(conv4)
            gmp = GlobalMaxPooling2D()(drop4)

            norm4 = BatchNormalization()(gmp)
            softmax = Dense(len(classes), activation='softmax')(norm4)
            # softmax = Dense(2, activation='softmax')(image_input)

            model = Model(inputs=image_input, outputs=softmax)

        elif model_type == 'efficient':
            freeze_convs = False
            feature_extractor = EfficientNetB2(
                include_top=False)
            if freeze_convs:
                for layer in feature_extractor.layers:
                    layer.trainable = False
            pooled_output = GlobalMaxPooling2D()(feature_extractor.output)
            output = Dense(len(classes), activation='softmax')(pooled_output)
            model = Model(
                inputs=feature_extractor.inputs,
                outputs=output)

        elif model_type == 'mobilenet':
            freeze_convs = False
            feature_extractor = MobileNetV2(
                include_top=False)
            if freeze_convs:
                for layer in feature_extractor.layers:
                    layer.trainable = False
            image_input = Input(shape=self.image_params['image_size'])
            feature_extractor_input = Concatenate(axis=3)((image_input, image_input, image_input))
            feature_extractor_output = feature_extractor(feature_extractor_input)
            pooled_output = GlobalMaxPooling2D()(feature_extractor_output)
            output = Dense(len(classes), activation='softmax')(pooled_output)
            model = Model(
                inputs=image_input,
                outputs=output)


        else:
            raise('Unrecognized model type requested.')

        # compile
        opt = optimizers.Adam(lr=0.0001)

        self.model = model
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


    # training
    def train(self, train_image_dir, validation_image_dir=None, n_epochs=10, batch_size=32, save_dir=None):

        mc = ModelCheckpoint(filepath=os.path.join(save_dir, 'model.{epoch:02d}'), monitor='val_loss',
                             verbose=1, period=5)
        tb = TensorBoard(log_dir=save_dir)
        # es = EarlyStopping(patience=2, monitor='val_acc', mode='max', baseline=0.93)

        # get the number of training and val examples
        n_train = len(glob.glob(os.path.join(validation_image_dir, '*_mask.tif')))
        n_val = len(glob.glob(os.path.join(validation_image_dir, '*_mask.tif')))

        image_params = self.image_params

        image_params['image_dir'] = train_image_dir

        train_image_generator = ClassificationDataGenerator(image_params,
                                              self.classes,  # labels need to match names of subfolders with images from that class
                                              self.labels,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              loss=self.loss)

        image_params['image_dir'] = validation_image_dir

        validation_image_generator = ClassificationDataGenerator(image_params,
                                                   self.classes,  # labels need to match names of subfolders with images from that class
                                                   self.labels,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   loss=self.loss)

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


# based on tensorflow example
def make_generator_model(image_size, n_channels=256, condensed_size=7, kernel_size=5):

    current_size = condensed_size
    model = Sequential()
    model.add(Dense(condensed_size*condensed_size*(n_channels), use_bias=False, input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((condensed_size, condensed_size, n_channels)))
    assert model.output_shape == (None, condensed_size, condensed_size, n_channels)  # Note: None is the batch size

    n_layers = 1
    model.add(Conv2DTranspose(n_channels, (kernel_size, kernel_size), strides=(1, 1), padding='same',
                              use_bias=False))
    assert model.output_shape == (None, current_size, current_size, n_channels)
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    while (current_size * 2) * 2 <= image_size:

        model.add(Conv2DTranspose(n_channels / 2 ** n_layers, (kernel_size, kernel_size), strides=(2, 2), padding='same', use_bias=False))
        current_size = current_size * 2
        assert model.output_shape == (None, current_size, current_size, n_channels / 2 ** n_layers)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        n_layers += 1

    model.add(Conv2DTranspose(1, (kernel_size, kernel_size), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, current_size * 2, current_size * 2, 1)

    return model


# based on tensorflow example
def make_discriminator_model(image_size, model_type='basic', n_channels=16):
    if model_type == 'basic':
        n_layers = 1
        current_size = image_size
        condensed_size = 7
        kernel_size = 5
        dropout = 0.3

        model = Sequential()

        while (current_size / 2) > condensed_size:
            if n_layers == 1:
                model.add(
                    Conv2D(n_channels * 2 ** (n_layers - 1), (kernel_size, kernel_size), strides=(2, 2), padding='same',
                           use_bias=False, input_shape=(image_size, image_size, 1)))
            else:
                model.add(
                    Conv2D(n_channels * 2 ** (n_layers - 1), (kernel_size, kernel_size), strides=(2, 2), padding='same',
                           use_bias=False))
            model.add(LeakyReLU())
            model.add(Dropout(dropout))
            model.add(BatchNormalization())

            current_size = current_size / 2
            n_layers += 1

        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))

    elif model_type == 'MobileNet':
        raise('MobileNet not yet implemented, exiting')
    else:
        raise('Unrecognized model type requested, exiting')

    return model
