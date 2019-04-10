import tensorflow as tf

from .discriminator import Disctriminator

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import layers

class TestDisctriminator(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)
        
        with tf.variable_scope("discriminator") as scope:
            self.bns = [layers.batch_norm(name="d_bn0{}".format(i,)) for i in range(2)]
            """
            m = tf.keras.Sequential()
            # 256x256x3
            #self.m.add(tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
            # 128x128x32
            #self.m.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            #self.m.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
            # 64x64x64
            #self.m.add(tf.keras.layers.BatchNormalization())
            #self.m.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            m.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=(image_size, image_size, 3)))
            # 32x32x128
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            m.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
            # 16x16x256
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.8))
            m.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
            # 8x8x256
            m.add(tf.keras.layers.LeakyReLU(alpha=0.8))

            # Some fully connected layers
            m.add(tf.keras.layers.Flatten())
            #m.add(tf.keras.layers.Dense(16))
            #m.add(tf.keras.layers.Dense(32))
            #m.add(tf.keras.layers.Dense(100))
            #m.add(tf.keras.layers.Dense(200))
            #m.add(tf.keras.layers.Dense(100))
            m.add(tf.keras.layers.Dense(50))
            m.add(tf.keras.layers.Dense(10))
            m.add(tf.keras.layers.Dense(1, name="out"))
            m.add(tf.keras.layers.Softmax(name="out_logits"))

            # Use named layers and create model here in order to get output from both two last layers
            self.model = tf.keras.Model(inputs=m.input, outputs=[m.get_layer("out_logits").output, m.get_layer("out").output])
            """

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            
            #64x64x3
            x0 = layers.lrelu(layers.conv2d(image, 128, k_h=3, k_w=3, d_h=2, d_w=2, name='d_00_conv'), leak=0.8)
            #32x32x128
            x1 = layers.lrelu(self.bns[0](layers.conv2d(x0, 256, k_h=3, k_w=3, d_h=2, d_w=2, name='d_01_conv'), is_training), leak=0.8)
            #16x16x256
            x2 = layers.lrelu(self.bns[1](layers.conv2d(x1, 256, k_h=3, k_w=3, d_h=2, d_w=2, name='d_02_conv'), is_training), leak=0.8)
            #8x8x256
            x3 = layers.linear(tf.reshape(x2, [-1, 8*8*256]), 32, 'd_03_lin')
            #32
            x4 = layers.linear(x3, 16, 'd_04_lin')
            #16
            x5 = layers.linear(x4, 1, 'd_05_lin')
            x6 = tf.nn.sigmoid(x5)
            return x6, x5

            #return self.model(image, is_training)