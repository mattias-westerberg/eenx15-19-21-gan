''' Import Packages '''
import numpy as np
import tensorflow as tf
import tensorboard
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
import cv2

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 960

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

    def __init__(self):
        self.model = tf.keras.Sequential()
        self.loss = 0
        self.build_generator()

    def up_sample(self, filter_size=0, stride=0, use_bias=False):
        # self.model.add(layers.MaxPool2D(pool_size=(2, 2), padding='valid', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
        # self.model.add(layers.Reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
        pass

    def down_sample(self, filter_size, stride, use_bias=False):
        # TODO: implement!
        pass

    def build_generator(self):
        # TODO:
        # model = tf.keras.Sequential()
        self.model.add(layers.Reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
        # self.up_sample()
        self.model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(3))
        self.model.summary()
        # return model

    def loss_function(self, real_output, fake_output):
        self.loss = tf.keras.losses.mean_squared_error(real_output, fake_output)


class Discriminator:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss = 0
        self.build_discriminator()

    def up_sample(self, filter_size, stride, use_bias=False):
        self.model.add(layers.Conv2D(12, filter_size, strides=stride, padding='same', use_bias=use_bias))

    def down_sample(self, filter_size, stride, use_bias=False):
        # TODO: implement!
        pass

    def build_discriminator(self):
        # TODO:
        # self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        # self.model.add(tf.keras.layers.LeakyReLU())
        # self.model.add(tf.keras.layers.Dropout(0.3))
        #
        # self.model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        # self.model.add(tf.keras.layers.LeakyReLU())
        # self.model.add(tf.keras.layers.Dropout(0.3))
        #
        # self.model.add(tf.keras.layers.Flatten())
        # self.model.add(tf.keras.layers.Dense(2))
        self.model.add(layers.Reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 3), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))


    def loss_function(self, real_output, fake_output):
        real_loss = self.loss_fun(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fun(tf.zeros_like(fake_output), fake_output)
        self.loss = real_loss + fake_loss


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


@tf.function
def train_step(night_images, day_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator.model(day_images) # Generate an image with the generator

        real_output = discriminator.model(night_images) # Discriminator output on real image
        fake_output = discriminator.model(generated_images) # Discriminator output on fake generator image

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


# train_step(night_image/255.0, day_image/255.0)
# print('Generator loss:', generator.loss, 'Discriminator loss:', discriminator.loss)
# train_step(night_image/255.0, day_image/255.0)
# print('Generator loss:', generator.loss, 'Discriminator loss:', discriminator.loss)

print(np.shape(night_image))
generated_img = generator.model.predict(night_image)
print(np.shape(generated_img))
generated_img /= 255.0

plt.figure()
plt.subplot(121)
plt.imshow(night_image[0]/255.0)
plt.title('Grounds Truth')
plt.subplot(122)
plt.imshow(generated_img[0])
plt.title('Generated image')

plt.show()

