''' Import Packages '''
import numpy as np
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import cv2
from ops import*
from glob import glob
from six.moves import xrange
import PIL

print(tf.__version__)

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

''' Load image '''


def load(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    image = np.array(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Shape of image = ", np.shape(image))
    image = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    image = np.reshape(image, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    return image


''' Create the generator '''


class Generator:

    def __init__(self, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3):
        self.model = tf.keras.Sequential()
        self.loss = 0
        self.build_generator()
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

    def up_sample(self, filter_size=0, stride=0, use_bias=False):
        # self.model.add(layers.MaxPool2D(pool_size=(2, 2), padding='valid', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
        # self.model.add(layers.Reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
        pass

    def down_sample(self, filter_size, stride, use_bias=False):
        # TODO: implement!
        pass

    def build_generator(self):
        # TODO:
        """ 256x256x3 """
        self.model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
        """ 128x128x64 """
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        """ 64x64x64 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        """ 32x32x128 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))

        """ 16x16x256 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same'))
        """ 32x32x128 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same'))
        """ 64x64x128 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same'))
        """ 128x128x64 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'))
        """ 256x256x32 """
        self.model.add(layers.Dense(3, activation='tanh'))

        self.model.summary()
        # return model

    def loss_function(self, real_output, fake_output):
        self.loss = tf.keras.losses.mean_squared_error(real_output, fake_output)

    # def generator(self, image, y=None):
    #     filter_channels = 3
    #     w = tf.get_variable('w', [5, 5, np.shape(image)[-1], filter_channels],
    #                         initializer=tf.truncated_normal_initializer(stddev=0.02))
    #     image = tf.nn.conv2d(image, w, [1, 2, 2, 1], padding='SAME')
    #
    #
    #     return image


class Discriminator:
    def __init__(self):
        self.model = tf.keras.Sequential()
        # self.loss_fun = tf.keras.losses.binary_crossentropy()
        self.loss = 0
        self.build_discriminator()

    def up_sample(self, filter_size, stride, use_bias=False):
        self.model.add(layers.Conv2D(12, filter_size, strides=stride, padding='same', use_bias=use_bias))

    def down_sample(self, filter_size, stride, use_bias=False):
        # TODO: implement!
        pass

    def build_discriminator(self):
        # TODO:
        """ 256x256x3 """
        self.model.add(layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
        """ 128x128x32 """
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        """ 64x64x64 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        """ 32x32x128 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        """ 16x16x256 """
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=0.2))
        self.model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        """ 8x8x256 """
        self.model.add(layers.LeakyReLU(alpha=0.8))
        """ Some fully connected layers"""
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(16))
        self.model.add(layers.Dense(32))
        self.model.add(layers.Dense(100))
        self.model.add(layers.Dense(200))
        self.model.add(layers.Dense(100))
        self.model.add(layers.Dense(50))
        self.model.add(layers.Dense(10))
        self.model.add(layers.Dense(1))
        self.model.add(layers.Softmax())
        self.model.summary()


    def loss_function(self, real_output, fake_output):
        # real_loss = self.loss_fun(tf.ones_like(real_output), real_output)
        # fake_loss = self.loss_fun(tf.zeros_like(fake_output), fake_output)
        # self.loss = real_loss + fake_loss
        self.loss = tf.keras.losses.mean_squared_error(real_output, fake_output)




''' Define optimizers '''
generator_optimizer = tf.keras.optimizers.Adam(0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)


''' Build the generator and discriminator '''
generator = Generator()

discriminator = Discriminator()


''' Save checkpoints '''
# checkpoints_dir = './checkpoints'
# checkpoints_prefix = os.path.join(checkpoints_dir, 'ckpt')
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=new_generator.model, discriminator=new_discriminator.model)


''' Define the training loop '''
EPOCHS = 100
num_examples_to_generate = 1

def train(LISA_dataset, night_dataset, EPOCHS = 100):
    """ Define the training with number of epochs for the whole datasets. """
    for epoch in EPOCHS:
        for day_image, night_image in zip(LISA_dataset, night_dataset):
            train_step(night_image, day_image)


# @tf.function
def train_step(night_images, day_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator.model.predict(day_images) # Generate an image with the generator

        real_output = discriminator.model.predict(night_images) # Discriminator output on real image
        fake_output = discriminator.model.predict(generated_images) # Discriminator output on fake generator image

        generator.loss_function(real_output, fake_output) # Calculate the disc loss between real/fake - image
        discriminator.loss_function(real_output, fake_output) # -- || -- Gan loss for real/fake - image

        gradients_of_generator = gen_tape.gradient(generator.loss, generator.model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(discriminator.loss, discriminator.model.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.model.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.model.trainable_variables))


## TESTING THE PROGRAM ##
night_image = load('C:\mattiasGit\eenx15-19-21-gan\GAN\\night2.jpg')
real_image = load('C:\mattiasGit\eenx15-19-21-gan\GAN\ground_truth.jpg')
day_image = load('C:\mattiasGit\eenx15-19-21-gan\GAN\day.png')


''' PIX2PIX TEST! '''
pix_night = generator.model.predict(night_image)
print(discriminator.model.predict(pix_night))

train_step(real_image, night_image)

# plt.figure()
# plt.subplot(121)
# plt.imshow(night_image[0]/255.0)
# plt.title('Grounds Truth')
# plt.subplot(122)
# plt.imshow(pix_night[0])
# plt.title('Generated image')
#
# plt.show()

