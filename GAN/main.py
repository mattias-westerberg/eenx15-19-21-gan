''' Import Packages '''
import numpy as np 
import tensorflow as tf 
import tensorboard
import matplotlib.pyplot as plt 
import pickle
import tensorflow.keras.layers as layers
import os



''' Load image ''' 

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image, channels=3)

  image = tf.cast(image, tf.float32)
  
  return image

# image_tensor = load_image('C:\mattiasGit\eenx15-19-21-gan\GAN\\test.png')
# print(np.shape(image_tensor))


''' Create the generator ''' 
class Generator:

  def __init__(self, input_image):
      self.input_image = input_image
      self.model = tf.keras.Sequential()
      self.loss = 0
      self.build_generator()

  def up_sample(self, filter_size, stride, use_bias = False):
      self.model.add(layers.Conv2DTranspose(64, filter_size, strides = stride, padding = 'same', use_bias = use_bias))
      

  def down_sample(self, filter_size, stride, use_bias = False):
      # TODO: implement!
      pass

  def build_generator(self):
      # TODO:
      pass

  def loss_function(self, real_output, fake_output):
      self.loss += tf.keras.losses.mean_squared_error(real_output, fake_output)
      

class Discriminator:
    def __init__(self, input_image):
        self.input_image = input_image
        self.model = tf.keras.Sequential()
        self.loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss = 0
        self.build_discriminator()
      
    def up_sample(self, filter_size, stride, use_bias = False):
        self.model.add(layers.Conv2DTranspose(64, filter_size, strides = stride, padding = 'same', use_bias = use_bias))
      

    def down_sample(self, filter_size, stride, use_bias = False):
        # TODO: implement!
        pass

    def build_discriminator(self):
        # TODO:
        pass

    def loss_function(self, real_output, fake_output):
        real_loss = self.loss_fun(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fun(tf.zeros_like(fake_output), fake_output)
        self.loss = real_loss + fake_loss

''' Define optimizers '''    
generator_optimizer = tf.keras.optimizers.Adam(0.001)

discriminator_optimizer = tf.keras.optimizers.Adam(0.001)

''' Save checkpoints '''
image = []
new_generator = Generator(image)
new_discriminator = Discriminator(image)
'''
checkpoints_dir = './checkpoints'
checkpoints_prefix = os.path.join(checkpoints_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
generator=new_generator.model, discriminator=new_discriminator.model)
'''

''' Define the training loop '''
EPOCHS = 100
num_examples_to_generate = 1
filename = 'C:\mattiasGit\eenx15-19-21-gan\GAN\\test.png'
seed = load(filename)
plt.imshow(seed)
plt.show()

# @tf.function
def train_step():
    pass
