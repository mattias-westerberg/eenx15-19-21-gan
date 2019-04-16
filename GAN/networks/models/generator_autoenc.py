# Generetor structure with a Feed Forward implementation for improving
# content and clarity in the image

from .generator import Generator
from .layers import *
import tensorflow as tf


class AutoEncoderGenerator(Generator):
    def __init__(self, image_size):
        Generator.__init__(self)
        self.image_size = image_size
        assert (image_size == 256)
        with tf.variable_scope("generator"):
            self.bns = [batch_norm(name="g_bn0{}".format(i, )) for i in range(12)]

    def __call__(self, image, is_training):
        """

        :param image: Image with size 256x256 with 3 feature maps
        :param is_training: Boolean if network parameters should be frozen (False) or changeable (True)
        :return: Image 256x256 as tensor
        """

        with tf.variable_scope("generator"):
            # IMPORTANT: Updates the batch size dynamically for the conv2d-layers
            # to not fail when we change the batch size during runtime

            self.batch_size = tf.shape(image)[0]

            # Image size = 256
            # Increase feature maps to 64

            # 256x256x3
            x = conv2d(image, 64, (3, 3), (1, 1), name='g_00_conv')
            x = lrelu(x, 0.8)

            # Same amount of feature maps (64), saves feature maps from block 1 to variable fm1
            # Down to 128x128

            # 256x256x64
            x = conv2d(x, 64, (3, 3), (2, 2), name='g_01_conv')
            x = self.bns[0](x, is_training)
            x = lrelu(x, 0.8)

            # TODO: Maybe add a pooling layer here instead of using strides=(2,2),
            #  to downscale to 128x128
            #  add FeedForward to block 4

            # End of block 1
            # ------------------------------------------------------ #
            # Image size = 128

            # Increase feature maps to 128

            # 128x128x64
            x = conv2d(x, 128, (3, 3), (1, 1), name='g_02_conv')
            x = self.bns[1](x, is_training)
            x = lrelu(x, 0.8)

            # Same amount of feature maps (128)
            # 128x128x128
            x = conv2d(x, 128, (3, 3), (1, 1), name='g_03_conv')
            x = self.bns[2](x, is_training)
            x = lrelu(x, 0.8)

            # Same amount of feature maps (128), saves feature maps from block 2 to variable fm2
            # Down to 64x64
            # 128x128x128
            x = conv2d(x, 128, (3, 3), (2, 2), name='g_04_conv')
            x = self.bns[3](x, is_training)
            x = lrelu(x, 0.8)

            # TODO: Maybe add a pooling layer here instead of using strides=(2,2),
            #  to downscale to 64x64
            #  add FeedForward to block 5

            # End of block 2
            # ------------------------------------------------------ #
            # Image size = 64

            # Same amount of feature maps (128)
            # 64x64x128
            x = conv2d(x, 128, (3, 3), (1, 1), name='g_05_conv')
            x = self.bns[4](x, is_training)
            x = lrelu(x, 0.8)

            # Increase feature maps to 256
            # 64x64x128
            x = conv2d(x, 256, (3, 3), (1, 1), name='g_06_conv')
            x = self.bns[5](x, is_training)
            x = lrelu(x, 0.8)

            # Down to 128 feature maps
            # Up to 128x128

            # 64x64x256
            x = deconv2d(x, [self.batch_size, 128, 128, 128], (3, 3), (2, 2), name='g_07_deconv')
            x = self.bns[6](x, is_training)
            x = lrelu(x, 0.8)

            # TODO: Maybe add a upConv layer here instead of using strides=(2,2),
            #  to upscale to 128x128

            # End of block 3
            # ------------------------------------------------------ #
            # Image size = 128

            # Same amount of feature maps (256)
            # MERGE LAYER: Half of the feature maps come from block 2 and the rest from block 3
            # 128x128x128
            x = deconv2d(x, [self.batch_size, 128, 128, 128], (3, 3), (1, 1), name='g_08_deconvMerge')
            x = self.bns[7](x, is_training)
            x = lrelu(x, 0.8)

            # Decrease feature maps to 128
            # 128x128x256
            x = deconv2d(x, [self.batch_size, 128, 128, 128], (3, 3), (1, 1), name='g_09_deconv')
            x = self.bns[8](x, is_training)
            x = lrelu(x, 0.8)

            # Down to 64 feature maps
            # Up to 256x256
            # 128x128x128
            x = deconv2d(x, [self.batch_size, 256, 256, 64], (3, 3), (2, 2), name='g_10_deconv')
            x = self.bns[9](x, is_training)
            x = lrelu(x, 0.8)

            # TODO: Maybe add a upConv layer here instead of using strides=(2,2),
            #  to upscale to 256x256

            # End of block 4
            # ------------------------------------------------------ #
            # Image size = 256

            # Same amount of feature maps (128)
            # MERGE LAYER: Half of the feature maps come from block 1 and the rest from block 4
            # 256x256x64
            x = deconv2d(x, [self.batch_size, 256, 256, 64], (3, 3), (1, 1), name='g_11_deconvMerge')
            x = self.bns[10](x, is_training)
            x = lrelu(x, 0.8)

            # Decrease feature maps to 64
            # 256x256x128
            x = deconv2d(x, [self.batch_size, 256, 256, 64], (3, 3), (1, 1), name='g_12_deconv')
            x = self.bns[11](x, is_training)
            x = lrelu(x, 0.8)

            # Down to 3 feature maps
            # Dense layer, Kernel=(1,1)
            # 256x256x64
            x = conv2d(x, 3, (1, 1), (1, 1), name='g_13_dense')
            x = lrelu(x, 0.8)

            # End of block 5
            # ------------------------------------------------------ #

        # 256x256x3
        return tf.nn.tanh(x)



