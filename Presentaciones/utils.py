from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

def process_image(file):
    image = Image.open(BytesIO(file))
    image = np.asarray(image.resize((150, 150)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 255.0
    image = np.vstack([image])
    return image

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2), 
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(512, activation='relu'), 
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])
    
    model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])    
    return model


animals = {
    0: 'Gato',
    1: 'Perro'
}