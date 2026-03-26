import tensorflow as tf

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1e-6) / (denominator + 1e-6)

def hybrid_loss(y_true, y_pred):
    # Combine BCE for pixel-wise accuracy and Dice for overlap optimization
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dl
