''' Import Packages '''
import numpy as np 
import tensorflow as tf 
import tensorboard
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import cv2


IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512


''' Load image ''' 

def load(image_file):  
  image = cv2.imread(image_file, cv2.IMREAD_COLOR)
  image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

  return image


''' Create the generator ''' 
class Generator:

  def __init__(self):
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
      model = tf.keras.Sequential()
      model.add(layers.Reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

      return model

  def loss_function(self, real_output, fake_output):
      self.loss += tf.keras.losses.mean_squared_error(real_output, fake_output)
      

class Discriminator:
    def __init__(self):
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
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))

        return model

    def loss_function(self, real_output, fake_output):
        real_loss = self.loss_fun(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fun(tf.zeros_like(fake_output), fake_output)
        self.loss = real_loss + fake_loss


''' Define optimizers '''
generator_optimizer = tf.keras.optimizers.Adam(0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)


new_generator = Generator()
test_generator = new_generator.build_generator()

new_discriminator = Discriminator()
test_discriminator = new_discriminator.build_discriminator()


''' Save checkpoints '''
#checkpoints_dir = './checkpoints'
#checkpoints_prefix = os.path.join(checkpoints_dir, 'ckpt')
#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=new_generator.model, discriminator=new_discriminator.model)


''' Define the training loop '''
EPOCHS = 100
num_examples_to_generate = 1

# @tf.function
def train_step(night_images, day_images):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = new_generator.model(day_images)

        real_output = new_discriminator.model(night_images)
        fake_output = new_discriminator.model(generated_images)

        gen_loss = new_generator.loss_function(real_output, fake_output)
        disc_loss = new_discriminator.loss_function(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, new_generator.model.trainable_variables) # OBS !!!
        gradients_of_discriminator = disc_tape.gradient(gen_loss, new_discriminator.model.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, new_generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, new_discriminator.trainable_variables))




## TESTING THE PROGRAM ##
night_image = load('./GAN/night.jpg')
real_image = load('./GAN/ground_truth.jpg')

night_image = np.reshape(night_image, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
generated_img = test_generator.predict(night_image)
generated_img /= 255.0
plt.imshow(generated_img[0])
plt.show()
