from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
import numpy as np
import util
from layers import *
import discriminator_util
import generator_util

"""
The parameters are:
sess:   the TensorFlow session to run in
image_size:   the width of the images, which should be the same as the height as we like square inputs
input_transform:   how to pre-process the input images
batch_size:   number of images to use in each run
sample_size:   number of z samples to take on each run, should be equal to batch_size
z_dim:   number of samples to take for each z
gf_dim:   dimension of generator filters in first conv layer
df_dim:   dimenstion of discriminator filters in first conv layer
gfc_dim:   dimension of generator units for fully-connected layer
dfc_gim:   dimension of discriminator units for fully-connected layer
c_dim:   number of image cannels (gray=1, RGB=3)
checkpoint_dir:   where to store the TensorFlow checkpoints
lam:  small constant weight for the sum of contextual and perceptual loss
"""
class GAN:
    def __init__(self, sess, image_size=64, input_transform=util.TRANSFORM_RESIZE, batch_size=64, sample_size=64,
                gf0_dim=64, gf1_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3, checkpoint_dir=None, lam=0.1, lowres_size=32):
        #image_size must be power of 2 and 8+
        assert(image_size & (image_size - 1) == 0 and image_size >= 8)

        self.sess = sess
        self.input_transform = input_transform
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]

        self.generator = generator_util.ImageGenerator(gf0_dim, gf1_dim, gfc_dim, image_size, batch_size)
        self.discriminator = discriminator_util.TestDisctriminator(df_dim, dfc_dim)

        self.lam = lam
        self.c_dim = c_dim
        self.lowres_size = lowres_size

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name="DCGAN.model"

    def build_model(self):
        """
        self.lowres_images = tf.reduce_mean(tf.reshape(self.images_real,
            [self.batch_size, self.lowres_size, self.lowres_size,
                self.lowres_size, self.lowres_size, self.c_dim]), [2, 4])
        self.lowres_G = tf.reduce_mean(tf.reshape(self.G,
            [self.batch_size, self.lowres_size, self.lowres_size,
                self.lowres_size, self.lowres_size, self.c_dim]), [2, 4])
        """

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.images_real = tf.placeholder(tf.float32, [None] + self.image_shape, name='images_real')
        self.images_input = tf.placeholder(tf.float32, [None] + self.image_shape, name='images_input')
        #self.images_input_sum = tf.summary.histogram("images_input", self.images_input)

        self.G = self.generator(self.images_input, self.is_training)
        
        self.D, self.D_logits = self.discriminator(self.images_real, is_training=self.is_training)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True, is_training=self.is_training)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        #IMPORTANT PREFIXES
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, config):
        data_input = util.get_paths(config.dataset_input)
        data_real = util.get_paths(config.dataset_real)
        np.random.shuffle(data_input)
        np.random.shuffle(data_real)

        assert(len(data_input) > 0 and len(data_real) > 0)
        
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
            
        #self.g_sum = tf.summary.merge([self.images_input_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        #self.d_sum = tf.summary.merge([self.images_input_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.g_sum = tf.summary.merge([self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        
        
        sample_files_input = data_input[0:self.sample_size]  
        sample_input = [util.get_image(sample_file, self.image_size, input_transform=self.input_transform) for sample_file in sample_files_input]
        sample_images_input = np.array(sample_input).astype(np.float32)
        
        sample_files_real = data_real[0:self.sample_size]  
        sample_real = [util.get_image(sample_file, self.image_size, input_transform=self.input_transform) for sample_file in sample_files_real]
        sample_images_real = np.array(sample_real).astype(np.float32)
        
        counter = 1
        start_time = time.time()
        
        if self.load(self.checkpoint_dir):
            print(""" An existing model was found - delete the directory or specify a new one with --checkpoint_dir """)
        else:
            print(""" No model found - initializing a new one""")
        
        for epoch in range(config.epoch):
            batch_idxs = min(len(data_input), config.train_size) // self.batch_size

            for idx in range(batch_idxs):
                batch_files_input = data_input[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_files_real = data_real[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_input = [util.get_image(batch_file, self.image_size, input_transform=self.input_transform) for batch_file in batch_files_input]
                batch_real = [util.get_image(batch_file, self.image_size, input_transform=self.input_transform) for batch_file in batch_files_real]
                batch_images_input = np.array(batch_input).astype(np.float32)
                batch_images_real = np.array(batch_real).astype(np.float32)
                
                #update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.images_real: batch_images_real, self.images_input: batch_images_input, self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                
                #update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.images_input: batch_images_input, self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                
                #run g_optim twice to make sure that d_loss does not go to zero (not in the paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.images_input: batch_images_input, self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({self.images_input: batch_images_input, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images_real: batch_images_real, self.is_training: False})
                errG = self.g_loss.eval({self.images_input: batch_images_input, self.is_training: False})
                
                counter += 1
                print("Epoch [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                        epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))
                
                if np.mod(counter, 3) == 1:
                    samples, d_loss, g_loss = self.sess.run([self.G, self.d_loss, self.g_loss], 
                                                            feed_dict={self.images_input: batch_images_real, self.images_real: batch_images_real, self.is_training: False})
                    util.save_images(samples, [8,8], './samples/train_{:02d}-{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
                    
                if np.mod(counter, 10) == 2:
                    self.save(config.checkpoint_dir, counter)


    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False