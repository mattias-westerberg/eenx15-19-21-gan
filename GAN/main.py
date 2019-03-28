
# https://mlnotebook.github.io/post/GAN5/
'''Import Packages'''
import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import TensorBoard
import cv2
import time
import os
from gan import GAN

def list_image_files(directory):
    extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
    paths = [f for f in os.listdir(directory) if any(f.endswith(ext) for ext in extensions)]
    return paths

def load_images(paths):
    images = []
    for p in paths:
        image = cv2.imread(p, cv2.IMREAD_COLOR)
        image = np.array(image, dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print('Shape of image = ', np.shape(image))
        image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        image = np.reshape(image, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        images.append(image)
    return images

'''Save checkpoints'''
# checkpoints_dir = './checkpoints'
# checkpoints_prefix = os.path.join(checkpoints_dir, 'ckpt')
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=new_generator.model, discriminator=new_discriminator.model)

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

EPOCHS = 100
num_examples_to_generate = 1

#night_images = load_images(list_image_files('test_images_night'))
#day_images = load_images(list_image_files('test_images_day'))

#DEFINE THE FLAGS FOR RUNNING SCRIPT FROM THE TERMINAL
# ARG1 = NAME OF THE FLAG
# ARG2 = DEFAULT VALUE
# ARG3 = DESCRIPTION
flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Number of epochs to train [20]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam optimiser [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term for adam optimiser [0.5]")
flags.DEFINE_integer("train_size", 100, "The size of training images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The batch-size (number of images to train at once) [64]")
flags.DEFINE_integer("image_size", 64, "The size of the images [n x n] [64]")
flags.DEFINE_string("dataset", "test_images_day","Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

#CREATE SOME FOLDERS FOR THE DATA
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
    
# GET ALL OF THE OPTIONS FOR TENSORFLOW RUNTIME
# Options to limit GPU usage
# Fixed the problem: tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True 
config.intra_op_parallelism_threads = 8

with tf.Session(config=config) as sess:
    #INITIALISE THE GAN BY CREATING A NEW INSTANCE OF THE DCGAN CLASS
    dcgan = GAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  is_crop=True, checkpoint_dir=FLAGS.checkpoint_dir)
    #TRAIN THE GAN
    dcgan.train(FLAGS)











#Options to limit GPU usage
#Fixed the problem: tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR

"""
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)

log_path = './logs'

NAME = 'AdamGAN-{}'.format(int(time.time()))

g = tf.Graph()

with g.as_default():
    '''Build the generator and discriminator '''
    generator = Generator((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    discriminator = Discriminator((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    gan = GAN(generator, discriminator)
    callback = TensorBoard(log_path)
    callback.set_model(discriminator)

    # ***NOTE global_variables_initializer() must be called before creating a tf.Session()!***
    init = tf.global_variables_initializer()
    cost = gan.discriminator.loss
    tf.summary.scalar('loss', cost)

writer = tf.summary.FileWriter(log_path + '/train', graph=g)
"""
#tensorboard = TensorBoard(log_dir='/logs/{}'.format(NAME))


# create a session for training and feedforward (prediction). Sessions are TF's way to run
# feed data to placeholders and variables, obtain outputs and update neural net parameters

"""
with tf.Session(graph=g) as sess:
    # ***initialization of all variables... NOTE this must be done before running any further sessions!***
    sess.run(init)
    #plt.figure()

    for i in tqdm.tqdm(range(100)):
        generated_img = gan.train_step(night_image, day_image)
        
        g_loss = int(gan.generator.loss.eval())
        d_loss = int(gan.discriminator.loss.eval())

        print('Generator loss: ' + g_loss)
        print('Discriminator loss: ' + d_loss)

        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_loss', d_loss)
        summaries = tf.summary.merge_all()
        summary = sess.run(summaries)

        #tf.summary.image('generated_image', generated_img)
        #tf.summary.scalar('discriminator_loss', discriminator.loss)
        #merged = tf.summary.merge_all()
        #summary = sess.run([generated_img, discriminator.loss])
        #writer.add_summary(summary)

        '''
        plt.subplot(121)
        plt.imshow(day_image[0])
        plt.title('Input image')

        plt.subplot(122)
        #plt.imshow(generated_img[0])
        image = sess.run(generated_img)[0]
        print(np.shape(image))
        plt.imshow(image.astype(np.uint8))
        plt.title('Generated image')

        plt.show()
        '''
"""