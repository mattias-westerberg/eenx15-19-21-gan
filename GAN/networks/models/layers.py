import numpy as np
import tensorflow as tf
import scipy.misc

#BATCH NORMALISATION OBJECT
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.99, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, training):
        with tf.variable_scope(self.name):
            return tf.layers.batch_normalization(x, momentum=self.momentum, epsilon=self.epsilon,
                                                center=True, scale=True, training=training)   

#CONVOLUTION FUNCTION
def conv2d(input_, output_dim, k, s, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k[0], k[1], input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, s[0], s[1], 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv   
    
#REVERSE CONVOLUTION FUNCTION
def deconv2d(input_, output_shape, k, s, stddev=0.02, name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k[0], k[0], output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                            strides=[1, s[0], s[1], 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv    
    
#NON-LINEARITY FUNCTION
def lrelu(x, leak, name="lrelu"):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x, leak, name)

def relu(x, name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(x, name)


def dropout(x, rate=0.5, name='dropout'):
    with tf.variable_scope(name):
        return tf.nn.dropout(x, rate=rate)

        
#LINEAR FUNCTION
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def flatten(x, output_dim):
    return tf.reshape(x, [-1, output_dim])

def max_pool(input_, k, s, name="MaxPool"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_, ksize=[1, k[0], k[1], 1], strides=[1, s[0], s[1], 1], padding='SAME', name=name)