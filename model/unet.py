from tensorflow.keras import layers, Model

def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(input_shape)
    
    # Simple Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Bottleneck
    bn = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    
    # Simple Decoder
    u1 = layers.UpSampling2D((2, 2))(bn)
    concat = layers.Concatenate()([u1, c1])
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c2) # [cite: 131]
    return Model(inputs, outputs)
