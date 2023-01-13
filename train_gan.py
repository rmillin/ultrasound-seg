"""
based on DCGAN example code from TF
"""

import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

from models import make_discriminator_model, make_generator_model
from custom_losses import generator_loss, discriminator_loss


def generate_and_save_images(model, epoch, test_input, output_image_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(output_image_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    # plt.show()


mean_generator_loss = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
mean_discriminator_loss = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)
# mean_discriminator_accuracy = tf.keras.metrics.Mean('discriminator_accuracy', dtype=tf.float32)


@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        # disc_acc = tf.add(tf.zeros_like(real_output) == tf.round(fake_output), tf.ones_like(real_output) == tf.round(real_output))/2

        mean_generator_loss(gen_loss)
        mean_discriminator_loss(disc_loss)
        # mean_discriminator_accuracy(disc_acc)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs, output_image_dir):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            if image_batch.shape[0] == batch_size:
                train_step(image_batch)

        # Produce images for the GIF as you go
        generate_and_save_images(generator,
                             epoch + 1,
                             seed,
                             output_image_dir)

        # Save the model every 15 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             seed,
                             output_image_dir)


image_size = 128
generator_lr = 1e-4
discriminator_lr = 1e-4
n_epochs = 100
noise_dim = 100
num_examples_to_generate = 16
buffer_size = 60000
batch_size = 4
n_generator_layers = 4
discriminator_model_type = 'basic'
image_dir = r'/Users/rmillin/Downloads/ultrasound-nerve-segmentation/train'

checkpoint_dir = '/Users/rmillin/Downloads/ultrasound-nerve-segmentation/results/debug/gan/training_checkpoints'
output_image_dir = '/Users/rmillin/Downloads/ultrasound-nerve-segmentation/results/debug/gan/images'

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

generator = make_generator_model(image_size, condensed_size=int(image_size/(2**n_generator_layers)))
discriminator = make_discriminator_model(image_size, model_type=discriminator_model_type)
generator_optimizer = tf.keras.optimizers.Adam(generator_lr)
discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_lr)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# get the list of label files, then for each find the matching image
all_label_files = glob.glob(os.path.join(image_dir, '*_mask.tif'))
# UNCOMMENT BELOW FOR TESTING
# all_label_files = all_label_files[:50]
all_image_files = [file.replace('_mask', '') for file in all_label_files]

train_images = np.zeros(((len(all_image_files),) + (image_size, image_size, 1)), dtype=np.float32)
for count, image_file in enumerate(all_image_files):
    print('loading image ' + str(count))
    try:
        tmp_image = cv2.imread(image_file, 0)
        tmp_image = cv2.resize(tmp_image, (image_size, image_size))
    except:
        print('Skipping ' + image_file)
        continue
    train_images[count, :, :, :] = np.expand_dims(np.expand_dims(tmp_image, axis=0), axis=-1)

train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

train(train_dataset, n_epochs, output_image_dir)