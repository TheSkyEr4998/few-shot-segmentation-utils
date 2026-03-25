import tensorflow as tf
from utils import dice_loss

def build_unet():
    inputs = tf.keras.Input((128,128,1))

    # Encoder
    c1 = tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    # Bottleneck
    b = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(p2)

    # Decoder
    u1 = tf.keras.layers.UpSampling2D()(b)
    u1 = tf.keras.layers.Concatenate()([u1, c2])
    c3 = tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(u1)

    u2 = tf.keras.layers.UpSampling2D()(c3)
    u2 = tf.keras.layers.Concatenate()([u2, c1])
    c4 = tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(u2)

    outputs = tf.keras.layers.Conv2D(1,1,activation='sigmoid')(c4)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss=dice_loss,
        metrics=['accuracy']
    )

    return model
