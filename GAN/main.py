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
      model.add(layers.Conv2D(256, (3, 3), activation = 'relu', input_shape=(48, 48, 3)))
      model.add(layers.Dense(64, activation = 'softmax'))

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

''' Load images for testing the program '''
# day_image = load('C:\mattiasGit\eenx15-19-21-gan\GAN\\test.png')
night_image = load('C:\mattiasGit\eenx15-19-21-gan\GAN\\night.jpg')
real_image = load('C:\mattiasGit\eenx15-19-21-gan\GAN\\ground_truth.jpg')

''' Define optimizers '''
generator_optimizer = tf.keras.optimizers.Adam(0.001)

discriminator_optimizer = tf.keras.optimizers.Adam(0.001)

''' Save checkpoints '''

new_generator = Generator()
test_generator = new_generator.build_generator()
# test_generator.summary()
new_discriminator = Discriminator()
test_discriminator = new_discriminator.build_discriminator()
print(np.shape(test_discriminator))

print(np.shape(night_image))

night_image = np.reshape(night_image/255.0, [-1, 48, 48, 3])
# night_image = night_image/255.0

generated_img = test_generator(night_image)

print(np.shape(generated_img))

plot_img = tf.cast(generated_img/255, tf.float32)
plt.imshow(plot_img)
plt.show()







'''
checkpoints_dir = './checkpoints'
checkpoints_prefix = os.path.join(checkpoints_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,
generator=new_generator.model, discriminator=new_discriminator.model)
'''

''' Define the training loop '''
EPOCHS = 100
num_examples_to_generate = 1



# plot_img = tf.cast(input_image/255, tf.float32)
# plt.imshow(plot_img)
# plt.show()

# @tf.function
def train_step(night_iamges, day_images):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = new_generator.model(day_images)

        real_output = new_discriminator.model(night_iamges)
        fake_output = new_discriminator.model(generated_images)

        gen_loss = new_generator.loss_function(real_output, fake_output)
        disc_loss = new_discriminator.loss_function(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, new_generator.model.trainable_variables) # OBS !!!
        gradients_of_discriminator = disc_tape.gradient(gen_loss, new_discriminator.model.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, new_generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, new_discriminator.trainable_variables))


