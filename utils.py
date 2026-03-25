import tensorflow as tf
import numpy as np

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )
    
    return bce + (1 - dice)
