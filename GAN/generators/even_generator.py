import tensorflow as tf

from .generator import Generator

class EvenGenerator(Generator):
    def __init__(self, image_size):
        Generator.__init__(self)
        
        self.image_size = image_size

        with tf.variable_scope("generator") as scope:
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)))

            self.model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.BatchNormalization())
            
            self.model.add(tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            self.model.add(tf.keras.layers.BatchNormalization())

            self.model.add(tf.keras.layers.Dense(3, activation='tanh'))

    def __call__(self, image, is_training=False):
        with tf.variable_scope("generator") as scope:
            return self.model(image, is_training) 