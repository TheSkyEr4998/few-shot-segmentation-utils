import tensorflow as tf

def build_unet():
    inputs = tf.keras.Input((128,128,1))

    c1 = tf.keras.layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(p1)

    u1 = tf.keras.layers.UpSampling2D()(c2)
    concat = tf.keras.layers.Concatenate()([u1, c1])

    outputs = tf.keras.layers.Conv2D(1,1,activation='sigmoid')(concat)

    return tf.keras.Model(inputs, outputs)
