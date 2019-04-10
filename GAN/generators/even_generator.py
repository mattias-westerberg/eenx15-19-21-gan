import tensorflow as tf

from .generator import Generator

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import layers

class EvenGenerator(Generator):
    def __init__(self, image_size):
        Generator.__init__(self)
        
        self.image_size = image_size

        with tf.variable_scope("generator") as scope:
            self.bns = [layers.batch_norm(name="g_bn0{}".format(i,)) for i in range(6)]
            """
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3)))

            self.model.add(tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=1.0))
            self.model.add(tf.keras.layers.BatchNormalization(momentum=0.99))
            
            self.model.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=1.0))
            self.model.add(tf.keras.layers.BatchNormalization(momentum=0.99))
            
            self.model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=1.0))
            self.model.add(tf.keras.layers.BatchNormalization(momentum=0.99))
            
            self.model.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=1.0))
            self.model.add(tf.keras.layers.BatchNormalization(momentum=0.99))
            
            self.model.add(tf.keras.layers.Conv2D(96, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=1.0))
            self.model.add(tf.keras.layers.BatchNormalization(momentum=0.99))
            
            self.model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=1.0))
            self.model.add(tf.keras.layers.BatchNormalization(momentum=0.99))

            self.model.add(tf.keras.layers.Dense(3, activation='tanh'))
            """

    def __call__(self, image, is_training=False):
        with tf.variable_scope("generator") as scope:
            x0 = layers.lrelu(layers.conv2d(image, 384, k_h=3, k_w=3, d_h=1, d_w=1, name='g_00_conv'), leak=0.8)
            x1 = layers.lrelu(self.bns[0](layers.conv2d(x0, 384, k_h=3, k_w=3, d_h=1, d_w=1, name='g_01_conv'), is_training), leak=0.8)
            x2 = layers.lrelu(self.bns[1](layers.conv2d(x1, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='g_02_conv'), is_training), leak=0.8)
            x3 = layers.lrelu(self.bns[2](layers.conv2d(x2, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='g_03_conv'), is_training), leak=0.8)
            x4 = layers.lrelu(self.bns[3](layers.conv2d(x3, 128, k_h=3, k_w=3, d_h=1, d_w=1, name='g_04_conv'), is_training), leak=0.8)
            x5 = layers.lrelu(self.bns[4](layers.conv2d(x4, 96, k_h=3, k_w=3, d_h=1, d_w=1, name='g_05_conv'), is_training), leak=0.8)
            x6 = layers.lrelu(self.bns[5](layers.conv2d(x5, 64, k_h=3, k_w=3, d_h=1, d_w=1, name='g_06_conv'), is_training), leak=0.8)
            x7 = layers.lrelu(layers.conv2d(x5, 3, k_h=3, k_w=3, d_h=1, d_w=1, name='g_07_conv'), leak=1.0)
            x8 = tf.nn.tanh(x7)
            return x8
            
            # Set training for batch normalization layers to work correctly during training
            # as well as inference mode such as when we produce samples.
            #return self.model(image, training=is_training) 