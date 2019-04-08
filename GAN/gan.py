from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf

from util import util
from generators.even_generator import EvenGenerator
from generators.suhren_generator import SuhrenGenerator
from generators.nordh_generator import NordhGenerator
from discriminators.nordh_discriminator import NordhDisctriminator
from discriminators.tf_discriminator import TFDisctriminator
from generators.tf_generator import TFGenerator

class GAN:
    """
    PARAMETERS
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
    """
    def __init__(self, sess, image_size=64, input_transform=util.TRANSFORM_RESIZE, batch_size=64, sample_size=64,
                gf0_dim=64, gf1_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3, sample_interval=16, checkpoint_interval=32, checkpoint_dir=None):
        # image_size must be power of 2 and 8+
        assert(util.is_pow2(image_size) and image_size >= 8)

        self.sess = sess
        self.input_transform = input_transform
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]
        self.c_dim = c_dim

        self.sample_interval = sample_interval
        self.checkpoint_interval = checkpoint_interval

        self.is_input_annotations = False

        #self.generator = TFGenerator(gf0_dim, gf1_dim, gfc_dim, image_size, batch_size)
        self.discriminator = TFDisctriminator(df_dim, dfc_dim)

        self.generator = EvenGenerator(image_size)
        #self.discriminator = NordhDisctriminator(image_size)
        print(self.generator.model.summary())
        #print(self.discriminator.model.summary())

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name="DCGAN.model"

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images_real = tf.placeholder(tf.float32, [None] + self.image_shape, name='images_real')
        self.images_input = tf.placeholder(tf.float32, [None] + self.image_shape, name='images_input')
        # (batch_size, num_bboxes, [x0, y0, x1, y1, c])
        self.bboxes = tf.placeholder(tf.int32, shape=(None, 5))
        
        self.G = self.generator(self.images_input, self.is_training)

        self.D_real, self.D_real_logits = self.discriminator(self.images_real, is_training=self.is_training)
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, reuse=True, is_training=self.is_training)

        self.d_real_sum = tf.summary.histogram("d_real", self.D_real)
        self.d_fake_sum = tf.summary.histogram("d_fake", self.D_fake)

        str_input = "g_input/"
        str_output = "g_output/"

        with tf.name_scope(None):
            with tf.name_scope(str_input):
                self.G_input_sum = tf.summary.image("g_input_image", self.images_input)
            with tf.name_scope(str_output):
                self.G_output_sum = tf.summary.image("g_output_image", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits,
                                                    labels=tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,
                                                    labels=tf.zeros_like(self.D_fake)))
        self.g_loss_image = 1.0 * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits,
                                                    labels=tf.ones_like(self.D_fake)))
        
        def loop_body(i, losses, bb_imgs_in, bb_imgs_out):
            img_in = self.images_input[i]
            img_out = self.G[i]
            bb = self.bboxes[i]
            img_in_cropped = tf.image.crop_to_bounding_box(img_in, bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0])
            img_out_cropped = tf.image.crop_to_bounding_box(img_out, bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0])
            loss = tf.losses.mean_squared_error(labels=img_in_cropped, predictions=img_out_cropped)
            losses = tf.concat([losses, [loss]], 0)
            size = tf.shape(bb_imgs_in)[1]
            img_in_cropped_resized = tf.image.resize_images(img_in_cropped, size=[size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
            img_out_cropped_resized = tf.image.resize_images(img_out_cropped, size=[size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=False)
            bb_imgs_in = tf.concat([bb_imgs_in, [img_in_cropped_resized]], 0)
            bb_imgs_out = tf.concat([bb_imgs_out, [img_out_cropped_resized]], 0)
            i = tf.add(i, 1)
            return [i, losses, bb_imgs_in, bb_imgs_out]

        def g_loss_function():
            # https://stackoverflow.com/questions/41233462/tensorflow-while-loop-dealing-with-lists
            losses = tf.Variable([])
            bb_size = 64
            bb_imgs_in = tf.zeros([0, bb_size, bb_size, self.c_dim])
            bb_imgs_out = tf.zeros([0, bb_size, bb_size, self.c_dim])
            i = tf.constant(0)
            loop_cond = lambda i, losses, bb_imgs_in, bb_imgs_out: tf.less(i, self.batch_size)
            [i, losses, bb_imgs_in, bb_imgs_out] = tf.while_loop(loop_cond, loop_body, [i, losses, bb_imgs_in, bb_imgs_out], shape_invariants=[i.get_shape(), tf.TensorShape([None]), tf.TensorShape([None, bb_size, bb_size, self.c_dim]), tf.TensorShape([None, bb_size, bb_size, self.c_dim])])
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(None):
                with tf.name_scope(str_input):
                    self.images_input_cropped = bb_imgs_in
                    self.G_input_cropped_sum = tf.summary.image("g_input_crop", self.images_input_cropped)
                with tf.name_scope(str_output):
                    self.images_output_cropped = bb_imgs_out
                    self.G_output_cropped_sum = tf.summary.image("g_output_crop", self.images_output_cropped)

            return tf.reduce_mean(losses)

        self.use_bboxes = tf.placeholder(tf.bool, name="use_bboxes")
        self.g_loss_bbox = 10.0 * tf.cond(self.use_bboxes, g_loss_function, lambda: 0.0)

        self.g_loss = self.g_loss_image + self.g_loss_bbox
        self.d_loss = self.d_loss_real + self.d_loss_fake
        
        self.g_loss_bbox_sum = tf.summary.scalar("g_loss_bbox", self.g_loss_bbox)
        self.g_loss_image_sum = tf.summary.scalar("g_loss_image", self.g_loss_image)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.saver = tf.train.Saver(max_to_keep=1)
    
    def train(self, config):
        data_real = util.get_paths(config.dataset_real)

        if os.path.isdir(config.dataset_input):  
            print("Input dataset is a directory")
            self.is_input_annotations = False
            data_input = util.get_paths(config.dataset_input)
            dict_input = {path : [0]*5 for path in data_input}
        else:
            print("Input dataset is an annotations .txt file")  
            self.is_input_annotations = True
            dict_input = util.load_data(config.dataset_input)
            data_input = list(dict_input.keys())
            dict_input = {key : val[0] for key, val in dict_input.items()}
            dict_input = util.resize_bounding_boxes(dict_input, self.image_size)

        np.random.shuffle(data_input)
        np.random.shuffle(data_real)

        assert(len(data_input) > 0 and len(data_real) > 0)
        
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        
        self.g_sum = tf.summary.merge([
            self.d_fake_sum,
            self.G_input_sum,
            self.G_output_sum,
            self.d_loss_fake_sum,
            self.g_loss_sum,
            self.g_loss_image_sum])

        if (self.is_input_annotations):
            self.g_sum = tf.summary.merge([
            self.g_sum,
            self.g_loss_bbox_sum,
            self.G_input_cropped_sum,
            self.G_output_cropped_sum])

        self.d_sum = tf.summary.merge([
            self.d_real_sum,
            self.d_loss_real_sum,
            self.d_loss_sum])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        
        sample_files_input = data_input[0:self.sample_size]
        sample_input = [util.get_image(sample_file, self.image_size, input_transform=self.input_transform) for sample_file in sample_files_input]
        sample_images_input = np.array(sample_input).astype(np.float32)
        
        sample_files_real = data_real[0:self.sample_size]
        sample_real = [util.get_image(sample_file, self.image_size, input_transform=self.input_transform) for sample_file in sample_files_real]
        sample_images_real = np.array(sample_real).astype(np.float32)
        
        sample_bboxes = np.array([dict_input[key] for key in sample_files_input]).astype(np.int32)

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
                
                batch_bboxes = np.array([dict_input[key] for key in batch_files_input]).astype(np.int32)

                #update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.images_real : batch_images_real, self.images_input : batch_images_input, self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                
                #update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.images_input : batch_images_input, self.bboxes : batch_bboxes, self.is_training : True, self.use_bboxes : self.is_input_annotations})
                self.writer.add_summary(summary_str, counter)
                
                #run g_optim twice to make sure that d_loss does not go to zero (not in the paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.images_input : batch_images_input, self.bboxes : batch_bboxes, self.is_training: True, self.use_bboxes : self.is_input_annotations})
                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({self.images_input : batch_images_input, self.is_training : False})
                errD_real = self.d_loss_real.eval({self.images_real : batch_images_real, self.is_training : False})
                errG = self.g_loss.eval({self.images_input : batch_images_input, self.bboxes : batch_bboxes, self.is_training : False, self.use_bboxes : self.is_input_annotations})
                
                counter += 1
                print("Epoch [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                        epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))
                
                if np.mod(counter, self.sample_interval) == 1:
                    samples, d_loss, g_loss = self.sess.run([self.G, self.d_loss, self.g_loss], 
                                                            feed_dict={self.images_input : sample_images_input, self.bboxes : sample_bboxes, self.images_real : sample_images_real, self.is_training : False, self.use_bboxes : self.is_input_annotations})
                    util.save_images(samples, [8,8], './samples/train_{:02d}-{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
                    
                if np.mod(counter, self.checkpoint_interval) == 2:
                    self.save(config.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        """Save the current state of the model to the checkpoint directory"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)

    def load(self, checkpoint_dir):
        """Load a model from the checkpoint directory if it exists"""
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False