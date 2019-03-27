import tensorflow as tf
import tensorflow.keras.layers as layers


def generator(z, out_channel_dim, is_train=True, alpha=0.2, keep_prob=0.5):
    """
        Create the generator network
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """
    with tf.variable_creator_scope('generator', reuse=(not is_train)):
        # First fully connected layer, 4x4x1024
        fc = layers.dense(z, 4 * 4 * 1024, use_bias=False)
        fc = tf.reshape(fc, (-1, 4, 4, 1024))
        bn0 = layers.batch_normalization(fc, training=is_train)
        lrelu0 = tf.maximum(alpha * bn0, bn0)
        drop0 = layers.dropout(lrelu0, keep_prob, training=is_train)

        # Deconvolution, 7x7x512
        conv1 = layers.conv2d_transpose(drop0, 512, 4, 1, 'valid', use_bias=False)
        bn1 = layers.batch_normalization(conv1, training=is_train)
        lrelu1 = tf.maximum(alpha * bn1, bn1)
        drop1 = layers.dropout(lrelu1, keep_prob, training=is_train)

        # Deconvolution, 14x14x256
        conv2 = layers.conv2d_transpose(drop1, 256, 5, 2, 'same', use_bias=False)
        bn2 = layers.batch_normalization(conv2, training=is_train)
        lrelu2 = tf.maximum(alpha * bn2, bn2)
        drop2 = layers.dropout(lrelu2, keep_prob, training=is_train)

        # Output layer, 28x28xn
        logits = layers.conv2d_transpose(drop2, out_channel_dim, 5, 2, 'same')

        out = tf.tanh(logits)

        return out

gen = generator('night.jpg', 3)