import tensorflow as tf
import numpy as np

# ===============================
# DICE COEFFICIENT (EVAL ONLY)
# ===============================
def dice_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)

    return (2 * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )

# ===============================
# COMBINED LOSS (FIXED)
# ===============================
def combined_loss(y_true, y_pred):
    # BCE → scalar
    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)

    dice = (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

    return bce + (1 - dice)
