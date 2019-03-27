# Generator model from Tensorflow's "dcGAN"

import tensorflow.keras.layers as layers
import tensorflow as tf


class Generator:
    def __init__(self):
        self.model = None
        self.generated_image = None

    def make_generator_model(self):
        """Currently input size is 100x100"""

        # Creating the model
        model = tf.keras.Sequential()
        # Creating 7, 7, channels=256 layer
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())  # Normalize the nodes values between 0-1 (instead of 0-255)
        model.add(layers.LeakyReLU())           # No negative values

        model.add(layers.Reshape((7, 7, 256)))
        # Simple assert to check resolution and color channels
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        # Strides=(1, 1) to keep the resolution
        # This layer only changes the color channels from 256 -> 128
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Strides > 1 to increase image size and (1, 1) for square ratio
        # Color channel=64
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # Stride (2, 2) to double the size
        # Output Color channel=3 for RGB
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 3)

        return self.model

    def generated_image(self, input_image):
        self.generated_image = self.model(input_image, training=False)


gen = Generator()
gen.make_generator_model()