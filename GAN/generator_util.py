from abc import ABC, abstractmethod
import tensorflow as tf
from layers import *
import math

class Generator:
    def __init__(self):
        super(Generator, self).__init__()

    @abstractmethod
    def __call__(self, image, reuse=False, is_training=False):
        pass

class ImageGenerator(Generator):
    def __init__(self, gf0_dim, gf1_dim, gfc_dim, image_size, batch_size):
        Generator.__init__(self)
        
        self.gf0_dim = gf0_dim   
        self.gf1_dim = gf1_dim   
        self.gfc_dim = gfc_dim
        self.image_size = image_size
        self.batch_size = tf.placeholder(tf.float32, batch_size, name='g_batch_size')

        self.bns0 = [batch_norm(name="g_bn0{}".format(i,)) for i in range(4)]

        log_size = int(math.log(image_size) / math.log(2))
        self.bns1 = [batch_norm(name='g_bn1{}'.format(i,)) for i in range(log_size)]

    def __call__(self, image, is_training=False):
        with tf.variable_scope("generator") as scope:
            x0 = lrelu(conv2d(image, self.gf0_dim, name='g_00_conv'))
            x1 = lrelu(self.bns0[0](conv2d(x0, self.gf0_dim*2, name='g_01_conv'), is_training))
            x2 = lrelu(self.bns0[1](conv2d(x1, self.gf0_dim*4, name='g_02_conv'), is_training))
            x3 = lrelu(self.bns0[2](conv2d(x2, self.gf0_dim*8, name='g_03_conv'), is_training))
            x4 = linear(tf.reshape(x3, [-1, 8192]), 32, 'g_04_lin')
            z = x4

            self.z_, self.h0_w, self.h0_b = linear(z, self.gf1_dim*8*4*4, 'g_h0_lin', with_w=True)

            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf1_dim * 8])
            hs[0] = tf.nn.relu(self.bns1[0](hs[0], is_training))
            
            i=1             #iteration number
            depth_mul = 8   #depth decreases as spatial component increases
            size=8          #size increases as depth decreases
            
            # IMPORTANT: Updates the batch size dynamically for the conv2d-layers
            # to not fail when we change the batch size during runtime
            self.batch_size = tf.shape(image)[0]

            while size < self.image_size:
                name='g_h{}'.format(i)
                with tf.variable_scope(name):
                    hs.append(None)
                    hs[i], _, _ = conv2d_transpose(hs[i-1], [self.batch_size, size, size, self.gf1_dim*depth_mul],
                                                    name=name, with_w=True)
                    hs[i] = tf.nn.relu(self.bns1[i](hs[i], is_training))
                    
                    i += 1
                    depth_mul //= 2
                    size *= 2
                    
                    hs.append(None)
                    name = 'g_h{}'.format(i)
                    hs[i], _, _ = conv2d_transpose(hs[i-1], [self.batch_size, size, size, 3], name=name, with_w=True)
                
            return tf.nn.tanh(hs[i])  

class TestGenerator(Generator):
    def __init__(self, z_dim, gf_dim, gfc_dim, image_out_size, batch_size):
        Generator.__init__(self)
        
        self.z_dim = z_dim
        self.gf_dim = gf_dim   
        self.gfc_dim = gfc_dim
        self.image_out_size = image_out_size
        self.batch_size = batch_size

        log_size = int(math.log(image_out_size) / math.log(2))
        self.g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]

    def __call__(self, z, is_training=False):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], is_training))
            
            i=1             #iteration number
            depth_mul = 8   #depth decreases as spatial component increases
            size=8          #size increases as depth decreases
            
            while size < self.image_out_size:
                name='g_h{}'.format(i)
                with tf.variable_scope(name):
                    hs.append(None)
                    hs[i], _, _ = conv2d_transpose(hs[i-1], [self.batch_size, size, size, self.gf_dim*depth_mul],
                                                    name=name, with_w=True)
                    hs[i] = tf.nn.relu(self.g_bns[i](hs[i], is_training))
                    
                    i += 1
                    depth_mul //= 2
                    size *= 2
                    
                    hs.append(None)
                    name = 'g_h{}'.format(i)
                    hs[i], _, _ = conv2d_transpose(hs[i-1], [self.batch_size, size, size, 3], name=name, with_w=True)
                
            return tf.nn.tanh(hs[i])  