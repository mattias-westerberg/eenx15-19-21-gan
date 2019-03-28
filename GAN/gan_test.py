import numpy as np
import tensorflow as tf
import tensorboard
import tensorflow.keras.layers as layers
from keras.utils import plot_model
import cv2

class Generator:
    def __init__(self, in_shape):
        '''
        self.model = tf.keras.Sequential()
        self.model.add(layers.Reshape(shape, input_shape=shape))
        self.model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(3))
        '''
        self.loss = tf.Variable(0., name="loss")

        self.input = layers.Input(shape=in_shape)
        self.G1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(self.input)
        self.G2 = layers.BatchNormalization()(self.G1)
        self.output = layers.Dense(3)(self.G2)

        self.model = tf.keras.models.Model(self.input, self.output)

    def loss_function(self, real_output, fake_output):
        self.loss = tf.keras.losses.mean_squared_error(real_output, fake_output)

class Discriminator:
    def __init__(self, in_shape):
        '''
        #https://www.tensorflow.org/alpha/tutorials/generative/dcgan
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=shape))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))
        self.loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        #self.model.summary()
        '''

        self.loss = tf.Variable(0., name="loss")
        
        #self.input = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(generator.output)
        self.input = layers.Input(shape=in_shape)
        self.D1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(self.input)
        self.D2 = layers.LeakyReLU()(self.D1)
        self.D3 = layers.Dropout(0.3)(self.D2)
        self.D4 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(self.D3)
        self.D5 = layers.LeakyReLU()(self.D4)
        self.D6 = layers.Dropout(0.3)(self.D5)
        self.D7 = layers.Flatten()(self.D6)
        self.output = layers.Dense(1)(self.D7)

        self.model = tf.keras.models.Model(self.input, self.output)

        self.loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss_function(self, real_output, fake_output):
        real_loss = self.loss_fun(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fun(tf.zeros_like(fake_output), fake_output)
        self.loss = real_loss + fake_loss

class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        
        #self.model = tf.keras.models.Model(inputs=[self.generator.input], outputs=[self.generator.output, self.discriminator.output])
        #self.model.summary()
        #plot_model(self.model, to_file='demo.png', show_shapes=True)

        self.generator_optimizer = tf.train.AdamOptimizer(0.0001)
        self.discriminator_optimizer = tf.train.AdamOptimizer(0.0001)

    def train_step(self, night_image, day_image):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_image = self.generator.model(day_image, training = True) # Generate an image with the generator

            real_output = self.discriminator.model(night_image, training = True) # Discriminator output on real image
            fake_output = self.discriminator.model(generated_image, training = True) # Discriminator output on fake generator image

            self.generator.loss_function(real_output, fake_output) # Calculate the disc loss between real/fake - image
            self.discriminator.loss_function(real_output, fake_output) # -- || -- self loss for real/fake - image

            gradients_of_generator = gen_tape.gradient(self.generator.loss, self.generator.model.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(self.discriminator.loss, self.discriminator.model.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.model.trainable_variables))
        
        return generated_image