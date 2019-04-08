import tensorflow as tf

from .discriminator import Disctriminator


class NordhDisctriminator(Disctriminator):
    def __init__(self, image_size):
        Disctriminator.__init__(self)

        with tf.variable_scope("discriminator") as scope:
            m = tf.keras.Sequential()
            """ 64x64x3 """
            m.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', input_shape=(image_size, image_size, 3)))
            """ 64x64x64 """
            m.add(tf.keras.layers.LeakyReLU(alpha=0.95))

            m.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
            """ 32x32x64 """
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.95))
            m.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
            """ 32x32x128 """
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.95))
            m.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
            ''' 16x16x128 '''
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.95))
            m.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
            """ 16x16x256 """
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.95))
            m.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
            """ 8x8x256 """
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.95))
            m.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
            ''' 8x8x512 '''
            m.add(tf.keras.layers.BatchNormalization())
            m.add(tf.keras.layers.LeakyReLU(alpha=0.95))
            m.add(tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
            ''' 4x4x512'''

            """ Some fully connected layers"""
            m.add(tf.keras.layers.Dense(1024))
            m.add(tf.keras.layers.LeakyReLU(alpha=0.95))
            m.add(tf.keras.layers.Dense(10))
            m.add(tf.keras.layers.Dense(1, name="out", activation='sigmoid'))
            m.add(tf.keras.layers.Softmax(name="out_logits"))
            # m.add(tf.keras.activations.sigmoid())

            # Use named layers and create model here in order to get output from both two last layers
            self.model = tf.keras.Model(inputs=m.input,
                                        outputs=[m.get_layer("out_logits").output, m.get_layer("out").output])

    def __call__(self, image, reuse=False, is_training=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            return self.model(image, is_training)