# script for training the binary model (duh)

# images need to be saved with the following directory structure:

# [training_dir]/effusion
# [training_dir]/no_effusion
# [training_dir]/consolidation
# [training_dir]/no_consolidation

# [validation_dir]/effusion
# [validation_dir]/no_effusion
# [validation_dir]/consolidation
# [validation_dir]/no_consolidation

import segmentation_model
import pickle
import os
import glob
import cv2
import numpy as np

from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # for training on gpu

# paths
# training_dir = r'L:\Research and Engineering\Data for Projects\DARPA Ultrasound\Data\rmillin\frame_classification\frames\train'
# validation_dir = r'L:\Research and Engineering\Data for Projects\DARPA Ultrasound\Data\rmillin\frame_classification\frames\val'
# save_dir = r'L:\Research and Engineering\Data for Projects\DARPA Ultrasound\Data\rmillin\frame_classification\results\custom4'
training_dir = r'/con_data/rmillin/frame_classification/frames/train'
validation_dir = r'/con_data/rmillin/frame_classification/frames/val'
save_dir = r'/con_data/rmillin/frame_classification/results/custom4/curated_adam'

# model parameters
model_type = 'custom'
use_mask = True
feature = 'effusion'
n_trainable_layers = 0
custom_loss = False

# training parameters
n_epochs = 10000
batch_size = 64
image_params = {'image_size': (1065, 720, 3)}

# create a mask for a spatial prior
if model_type == 'custom':
    # load a few sample frames and sum them
    example_files = glob.glob(os.path.join(validation_dir, feature, '*.jpg'))
    samples = np.linspace(0, len(example_files)-10, 15).astype(int)
    for sample in samples:
        example_file = example_files[sample]
        example_frame = cv2.imread(example_file)
        if not sample:
            shp = example_frame.shape
            all_frames = np.expand_dims(example_frame[:, :, 0], axis=0)
        else:
            example_frame = cv2.resize(example_frame[:, :, 0], (shp[1], shp[0]))
            all_frames = np.concatenate((all_frames, np.expand_dims(example_frame, axis=0)), axis=0)
    sum_frame = np.sum(all_frames, axis=0)
    # spatial prior is zero where the image is zero, otherwise high in the center, low at the top and bottom
    _, y = np.meshgrid(np.arange(shp[1]), np.arange(shp[0]))
    mask = np.exp(-(y - shp[0]*4/9)**2/(shp[0]/2)**2)
    mask = np.multiply(mask, sum_frame > 0)
    mask = cv2.resize(mask, (14, 14))
    # set the top row to zero, since it is perpetually a problem
    mask[0, :] = 0
    mask = mask/np.max(mask)
    mask = mask * 0.99  # avoid any value being 1, for purposes of regularization
    mask = np.expand_dims(mask, axis=3)
    if custom_loss:
        mask = np.tile(mask, [1, 1, 128])
    else:
        mask = np.expand_dims(mask, axis=0)
        mask = np.tile(mask, [batch_size, 1, 1, 128])
    mask = mask.astype(np.float32)
    np.save(os.path.join(save_dir, 'mask.npy'), mask)
else:
    mask = None

# check if a model already exists; if so, load it; if not, create it
bin_model = segmentation_model.BinaryModel(model_type, n_trainable_layers, feature,
                              image_params, mask=mask, custom_loss=custom_loss)
try:
    model_checkpoint = glob.glob(os.path.join(save_dir, 'model*.hdf5'))[-1]
    keras_model = load_model(os.path.join(save_dir, model_checkpoint))
    bin_model.model = keras_model
    print('Loaded existing model to continue training.')
except:
    print('No existing model found; training from scratch.')

# train the model
bin_model.train(training_dir,
               validation_image_dir=validation_dir,
               n_epochs=n_epochs,
               batch_size=batch_size,
               save_dir=save_dir)

# save the model class instance
with open(os.path.join(save_dir, 'binary_model_instance.pickle'), 'wb') as f:
    pickle.dump(bin_model, f)