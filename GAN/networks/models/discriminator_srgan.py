import tensorflow as tf

from .discriminator import Disctriminator
from .layers import *

class SRGANDisctriminator(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)
        assert(image_size == 256)
        with tf.variable_scope("discriminator") as scope:
            self.bns = [batch_norm(name="d_bn0{}".format(i,)) for i in range(7)]

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            x = conv2d(image, 64, (3, 3), (1, 1), name='d_00_conv')
            x = lrelu(x, 0.2)

            x = conv2d(x, 128, (3, 3), (2, 2), name='d_01_conv')
            x = lrelu(x, 0.2)
            x = self.bns[0](x, is_training)

            # 128x128x128
            x = conv2d(x, 128, (3, 3), (2, 2), name='d_02_conv')
            x = lrelu(x, 0.2)
            x = self.bns[1](x, is_training)

            # 64x64x128
            x = conv2d(x, 256, (3, 3), (2, 2), name='d_03_conv')
            x = lrelu(x, 0.2)
            x = self.bns[2](x, is_training)

            # 32x32x128
            x = conv2d(x, 256, (3, 3), (2, 2), name='d_04_conv')
            x = lrelu(x, 0.2)
            x = self.bns[3](x, is_training)

            # 16x16x256
            x = conv2d(x, 256, (3, 3), (2, 2), name='d_05_conv')
            x = lrelu(x, 0.2)
            x = self.bns[4](x, is_training)

            # 8x8x256
            x = conv2d(x, 512, (2, 2), (2, 2), name='d_06_conv')
            x = lrelu(x, 0.2)
            x = self.bns[5](x, is_training)

            # 4x4x512
            x = conv2d(x, 1024, (1, 1), (2, 2), name='d_07_conv')
            x = lrelu(x, 0.2)
            x = self.bns[6](x, is_training)

            # 2x2x1024
            x = flatten(x)
            x = linear(x, 100, "d_08_lin")

            # 100
            x = linear(x, 10, "d_09_lin")

            # 10
            x = linear(x, 1, "d_10_lin")

            return x
