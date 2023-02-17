import numpy as np
import tensorflow as tf


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
        epsilon = 0.000001
        w = tf.keras.backend.constant(np.ones(size))
        w = tf.keras.backend.maximum(w, y_true * 5)  # make weights for positive entries higher
        w = w / tf.keras.backend.maximum(tf.keras.backend.sum(w), epsilon)
        # element-wise cross-entropy
        e_ce = y_true * tf.keras.backend.log(tf.keras.backend.maximum(y_pred, epsilon)) + (1 - y_true) * tf.keras.backend.log(tf.keras.backend.maximum(1 - y_pred, epsilon))
        loss = -tf.keras.backend.sum(e_ce * w)
        return loss

    return weighted_loss


def dice_loss():
    """
    dice loss
    :param y: true label
    :param y_pred: predicted label
    :return: loss
    """

    def compute_dice_loss(y_true, y_pred):
        loss = 1 - (tf.keras.backend.sum(y_true * y_pred) * 2 + 1) / \
               (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + 1)
        return loss

    return compute_dice_loss

# below from Tensorflow GAN example

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output, from_logits=True)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output, from_logits=True)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)

