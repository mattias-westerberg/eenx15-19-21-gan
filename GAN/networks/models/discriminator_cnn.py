import tensorflow as tf

from .discriminator import Disctriminator
from .layers import *

class CNNDisctriminator(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)
        #with tf.variable_scope("discriminator") as scope:
        #    with tf.variable_scope("cnn") as scope:
        #        self.d_bns = [batch_norm(name="d_bn{}".format(i,)) for i in range(3)]
        self.fm = 32
        self.image_size = image_size

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            with tf.variable_scope("cnn") as scope:
                if reuse:
                    scope.reuse_variables()

                #256x256x3
                x = conv2d(image, self.fm, (3, 3), (1, 1), name='d_h00_conv')
                x = lrelu(x, 0.2)

                #256x256xfm
                x = conv2d(x, self.fm * 2, (3, 3), (2, 2), name='d_h01_conv')
                x = lrelu(x, 0.2)
                x = tf.layers.batch_normalization(x, training=is_training, name='d_bns0')   
                #x = self.d_bns[0](x, is_training)

                #128x128x2fm
                x = conv2d(x, self.fm * 4, (3, 3), (2, 2), name='d_h02_conv')
                x = lrelu(x, 0.2)
                x = tf.layers.batch_normalization(x, training=is_training, name='d_bns1')   
                #x = self.d_bns[1](x, is_training)

                #64x64x4fm
                x = conv2d(x, self.fm * 8, (3, 3), (2, 2), name='d_h03_conv')
                x = lrelu(x, 0.2)
                x = tf.layers.batch_normalization(x, training=is_training, name='d_bns2')   
                #x = self.d_bns[2](x, is_training)

                #32x32x8fm
                x = flatten(x)
                x = linear(x, 32, 'd_h04_lin')

                #32
                x = linear(x, 1, 'd_h05_lin')

                #1
                return x