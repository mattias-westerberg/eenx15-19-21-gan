# https://mlnotebook.github.io/post/GAN5/
import tensorflow as tf
from gan import GAN

# Disable some TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DEFINE THE FLAGS FOR RUNNING SCRIPT FROM THE TERMINAL
# (ARG1, ARG2, ARG3) = (NAME OF THE FLAG, DEFAULT VALUE, DESCRIPTION)
flags = tf.app.flags
flags.DEFINE_string("model_name", "GAN", "The identifying name string of the model [GAN]")
flags.DEFINE_integer("epoch", 200, "Number of epochs to train [20]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam optimiser [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term for adam optimiser [0.5]")
flags.DEFINE_integer("train_size", 1000, "The size of training images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The batch-size (number of images to train at once) [64]")
flags.DEFINE_integer("image_size", 256, "The size of the images [n x n] [64]")
flags.DEFINE_string("dataset_real", "tests/test_images_night", "Real dataset directory [tests/test_images_night].")
flags.DEFINE_string("dataset_input", "tests/train_lisa_extension_20_12_crop_256.txt", "Input dataset directory [tests/test_images_day].")
flags.DEFINE_string("output_dir", "outputs", "Directory name to save the outputs of the model [outputs]")
flags.DEFINE_integer("sample_interval", 10, "Number of epochs between samples [16]")
flags.DEFINE_integer("sample_size", 16, "Number of samples to generate [16]")
flags.DEFINE_integer("checkpoint_interval", 10, "Number of epochs between checkpoints [32]")
flags.DEFINE_float("bbox_weight", 1.0, "Weight for the bbox content loss [1.0]")
flags.DEFINE_float("image_weight", 1.0, "Weight for the image content loss [1.0]")
FLAGS = flags.FLAGS

# Options to limit GPU usage
# Fixed the problem: tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.allow_growth = True 
config.intra_op_parallelism_threads = 8

with tf.Session(config=config) as sess:
    dcgan = GAN(sess, model_name=FLAGS.model_name, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  sample_size=FLAGS.sample_size, input_transform="resize", output_dir=FLAGS.output_dir, sample_interval=FLAGS.sample_interval, checkpoint_interval=FLAGS.checkpoint_interval, bbox_weight=FLAGS.bbox_weight, image_weight=FLAGS.image_weight)
    dcgan.train(FLAGS)