import tensorflow as tf

from .discriminator import Disctriminator
from .layers import *

class TFDisctriminator(Disctriminator):
    def __init__(self, df_dim, dfc_dim):
        Disctriminator.__init__(self)
        self.df_dim = df_dim
        self.dfc_dim = dfc_dim
        self.d_bns = [layers.batch_norm(name="d_bn{}".format(i,)) for i in range(4)]

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = layers.lrelu(layers.conv2d(image, self.df_dim, name='d_h00_conv'))
            h1 = layers.lrelu(self.d_bns[0](layers.conv2d(h0, self.df_dim*2, name='d_h1_conv'), is_training))
            h2 = layers.lrelu(self.d_bns[1](layers.conv2d(h1, self.df_dim*4, name='d_h2_conv'), is_training))
            h3 = layers.lrelu(self.d_bns[2](layers.conv2d(h2, self.df_dim*8, name='d_h3_conv'), is_training))
            h4 = layers.linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4