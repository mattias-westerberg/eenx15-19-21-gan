import tensorflow as tf

from .generator import Generator
from .layers import *


class EvenGenerator(Generator):
    def __init__(self, image_size):
        Generator.__init__(self)
        self.image_size = image_size
        assert(image_size == 256)
        #with tf.variable_scope("generator"):
        #    with tf.variable_scope("even"):
        #        self.bns = [batch_norm(name="g_bn0{}".format(i,)) for i in range(5)]

    def __call__(self, image, is_training=False):
        with tf.variable_scope("generator"):
            with tf.variable_scope("even") as scope:
                
                x = conv2d(image, 256, (3, 3), (1, 1), name='g_00_conv')
                x = lrelu(x, 0.2)

                x = conv2d(x, 128, (3, 3), (1, 1), name='g_01_conv')
                x = lrelu(x, 0.2)
                x = tf.layers.batch_normalization(x, training=is_training, name='g_bns0')   
                #x = self.bns[0](x, is_training)

                x = conv2d(x, 96, (3, 3), (1, 1), name='g_02_conv')
                x = lrelu(x, 0.2)
                x = tf.layers.batch_normalization(x, training=is_training, name='g_bns1')   
                #x = self.bns[1](x, is_training)

                x = conv2d(x, 64, (3, 3), (1, 1), name='g_03_conv')
                x = lrelu(x, 0.2)
                x = tf.layers.batch_normalization(x, training=is_training, name='g_bns2')   
                #x = self.bns[2](x, is_training)

                x = conv2d(x, 32, (3, 3), (1, 1), name='g_04_conv')
                x = lrelu(x, 0.2)
                x = tf.layers.batch_normalization(x, training=is_training, name='g_bns3')   
                #x = self.bns[3](x, is_training)

                x = conv2d(x, 16, (3, 3), (1, 1), name='g_05_conv')
                x = lrelu(x, 0.2)
                x = tf.layers.batch_normalization(x, training=is_training, name='g_bns4')   
                #x = self.bns[4](x, is_training)

                # Dense layer, kernel=(1, 1)
                x = conv2d(x, 3, (1, 1), (1, 1), name='g_06_conv')

                return tf.nn.sigmoid(x)