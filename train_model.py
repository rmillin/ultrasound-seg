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

from keras.models import load_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # for training on gpu

# paths
training_dir = r'/Users/rmillin/Documents/ultrasound-nerve-segmentation/train'
validation_dir = r'/Users/rmillin/Documents/ultrasound-nerve-segmentation/val'
save_dir = r'/Users/rmillin/Documents/ultrasound-nerve-segmentation/results/debug'

# model parameters
model_type = 'basic'
use_mask = True
classes = ['background', 'nerve']
labels = [0, 1]
custom_loss = False

# training parameters
n_epochs = 5
batch_size = 8
image_params = {'image_size': (512, 512, 1)}

# check if a model already exists; if so, load it; if not, create it
bin_model = segmentation_model.BinaryModel(image_params, model_type, classes, labels, custom_loss)

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
