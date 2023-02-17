import models
import pickle
import os
import glob

from tensorflow.keras.models import load_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # for training on gpu
os.environ["KERAS_BACKEND"] = "tensorflow"

# paths
training_dir = r'/Users/rmillin/Downloads/ultrasound-nerve-segmentation/train'
validation_dir = r'/Users/rmillin/Downloads/ultrasound-nerve-segmentation/val'
save_dir = r'/Users/rmillin/Downloads/ultrasound-nerve-segmentation/results/debug/dice'

# model parameters
model_task = 'segmentation' # classification or segmentation
model_type = 'basic'
classes = ['background', 'nerve']
labels = [0, 1]
loss = 'dice'

# training parameters
n_epochs = 20
batch_size = 4
# batch_size = 16
image_params = {'image_size': (128, 128, 1)}

# check if a model already exists; if so, load it; if not, create it
if model_task == 'classification':
    bin_model = models.ClassificationModel(image_params, model_type, classes, labels, loss, batch_size=batch_size)
elif model_task == 'segmentation':
    bin_model = models.SegmentationModel(image_params, model_type, classes, labels, loss, batch_size=batch_size)
else:
    raise('Unrecognized model task, exiting')


try:
    model_checkpoint = glob.glob(os.path.join(save_dir, '*.h5'))[-1]
    keras_model = load_model(model_checkpoint)
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
