import tensorflow as tf

from .discriminator import Disctriminator
from .layers import *

def relu(x):
    return tf.nn.relu(x)

# Usually handles 224x224x3 images
class Discriminator_VGG19(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)
        self.d_bns = [batch_norm(name="d_bn{}".format(i,)) for i in range(3)]
        self.fm = 32
        self.image_size = image_size

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            #256x256x3
            x = tf.layers.conv2d(image, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            #128x128x64
            x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            #64x64x128
            x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            #32x32x256
            x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            #16x16x512
            x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            #8x8x512
            x = tf.layers.dense(x, 4096, activation=relu)
            x = tf.layers.dropout(x, rate=0.5)
            x = tf.layers.dense(x, 4096, activation=relu)
            x = tf.layers.dropout(x, rate=0.5)

            
            x = flatten(x, 4096)
            x = linear(x, 1, 'l_out')

            return x