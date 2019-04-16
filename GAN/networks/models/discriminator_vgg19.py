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
        self.fm = 64
        self.image_size = image_size

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # 256x256x3
            x = conv2d(image, self.fm, (3, 3), (1, 1), name='vgg_00_conv')
            x = relu(x)

            x = conv2d(x, self.fm, (3, 3), (1, 1), name='vgg_01_conv')
            x = relu(x)
            x = max_pool(x, (2, 2), (2, 2), name='vgg_02_maxpool')


            # #256x256x3
            # x = tf.layers.conv2d(image, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            # 128x128x64
            x = conv2d(x, self.fm*2, (3, 3), (1, 1), name='vgg_03_conv')
            x = relu(x)
            x = max_pool(x, (2, 2), (2, 2), name='vgg_04_maxpool')

            # #128x128x64
            # x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            # 64x64x128
            x = conv2d(x, self.fm*4, (3, 3), (1, 1), name='vgg_04_conv')
            x = relu(x)
            x = conv2d(x, self.fm*4, (3, 3), (1, 1), name='vgg_05_conv')
            x = relu(x)
            x = conv2d(x, self.fm*4, (3, 3), (1, 1), name='vgg_06_conv')
            x = relu(x)
            x = conv2d(x, self.fm*4, (3, 3), (1, 1), name='vgg_07_conv')
            x = relu(x)
            x = max_pool(x, (2, 2), (2, 2), name='vgg_08_maxpool')

            #64x64x128
            # x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            #32x32x256
            x = conv2d(x, self.fm * 8, (3, 3), (1, 1), name='vgg_08_conv')
            x = relu(x)
            x = conv2d(x, self.fm * 8, (3, 3), (1, 1), name='vgg_09_conv')
            x = relu(x)
            x = conv2d(x, self.fm * 8, (3, 3), (1, 1), name='vgg_10_conv')
            x = relu(x)
            x = conv2d(x, self.fm * 8, (3, 3), (1, 1), name='vgg_11_conv')
            x = relu(x)
            x = max_pool(x, (2, 2), (2, 2), name='vgg_12_maxpool')

            #32x32x256
            # x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            x = conv2d(x, self.fm * 8, (3, 3), (1, 1), name='vgg_13_conv')
            x = relu(x)
            x = conv2d(x, self.fm * 8, (3, 3), (1, 1), name='vgg_14_conv')
            x = relu(x)
            x = conv2d(x, self.fm * 8, (3, 3), (1, 1), name='vgg_15_conv')
            x = relu(x)
            x = conv2d(x, self.fm * 8, (3, 3), (1, 1), name='vgg_16_conv')
            x = relu(x)
            x = max_pool(x, (2, 2), (2, 2), name='vgg_17_maxpool')

            #16x16x512
            # x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=relu)
            # x = tf.layers.max_pooling2d(x, (2, 2), (2, 2))

            x = linear(x, 4096, scope='vgg_18_dense')       # OBS: for linear layer name=scope instead
            x = dropout(x, rate=0.5, name='vgg_19_dropout')
            x = linear(x, 4096, scope='vgg_20_dense')
            x = dropout(x, rate=0.5, name='vgg_21_dropout')
            x = linear(x, 1, scope='vgg_22_dense')
            x = tf.nn.sigmoid(x)

            #8x8x512
            # x = tf.layers.dense(x, 4096, activation=relu)
            # x = tf.layers.dropout(x, rate=0.5)
            # x = tf.layers.dense(x, 4096, activation=relu)
            # x = tf.layers.dropout(x, rate=0.5)
            # x = tf.layers.dense(x, 1, activation='sigmoid')

            return x
