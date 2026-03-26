# FIX: Overwrite the metrics file with the high-performance version
metrics_content = """
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    # Thresholding at 0.5 for the final score
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - (numerator + 1e-6) / (denominator + 1e-6)

def hybrid_loss(y_true, y_pred):
    # BCE handles the pixels, Dice handles the shape overlap
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dl
"""

with open('/content/few-shot-segmentation-utils/utils/metrics.py', 'w') as f:
    f.write(metrics_content)

# Now clear the cache so Python sees the new file
import sys
if 'utils.metrics' in sys.modules:
    del sys.modules['utils.metrics']

from utils.metrics import dice_coef, hybrid_loss
print("✅ metrics.py repaired and upgraded with Hybrid Loss!")
