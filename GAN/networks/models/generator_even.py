import tensorflow as tf

from .generator import Generator
from .layers import *

class EvenGenerator(Generator):
    def __init__(self, image_size):
        Generator.__init__(self)
        self.image_size = image_size
        assert(image_size == 256)
        with tf.variable_scope("generator"):
            self.bns = [batch_norm(name="g_bn0{}".format(i,)) for i in range(6)]

    def __call__(self, image, is_training):
        with tf.variable_scope("generator"):

            x = conv2d(image, 64, (3, 3), (1, 1), name='g_00_conv')
            x = lrelu(x, 0.8)

            x = conv2d(x, 128, (3, 3), (1, 1), name='g_01_conv')
            x = self.bns[0](x, is_training)
            x = lrelu(x, 0.8)

            x = conv2d(x, 128, (3, 3), (1, 1), name='g_02_conv')
            x = self.bns[1](x, is_training)
            x = lrelu(x, 0.8)

            x = conv2d(x, 256, (3, 3), (1, 1), name='g_03_conv')
            x = self.bns[2](x, is_training)
            x = lrelu(x, 0.8)

            x = conv2d(x, 256, (3, 3), (1, 1), name='g_04_conv')
            x = self.bns[3](x, is_training)
            x = lrelu(x, 0.8)

            x = conv2d(x, 128, (3, 3), (1, 1), name='g_05_conv')
            x = self.bns[4](x, is_training)
            x = lrelu(x, 0.8)

            x = conv2d(x, 128, (3, 3), (1, 1), name='g_06_conv')
            x = self.bns[5](x, is_training)
            x = lrelu(x, 0.8)

            x = conv2d(x, 128, (3, 3), (1, 1), name='g_07_conv')
            x = self.bns[6](x, is_training)
            x = lrelu(x, 0.8)

            x = conv2d(x, 96, (3, 3), (1, 1), name='g_08_conv')
            x = self.bns[7](x, is_training)
            x = lrelu(x, 0.8)

            x = conv2d(x, 64, (3, 3), (1, 1), name='g_09_conv')
            x = self.bns[8](x, is_training)
            x = lrelu(x, 0.8)

            # Dense layer, kernel=(1, 1)
            x = conv2d(x, 3, (1, 1), (1, 1), name='g_07_conv')
            x = lrelu(x, 1.0)

            return tf.nn.tanh(x)
