from abc import ABC, abstractmethod

class GAN:
    def __init__(self):
        super(GAN, self).__init__()

    @abstractmethod
    def __call__(self, image, reuse=False, is_training=False):
        pass

    @abstractmethod
    def train(FLAGS):
        pass

    def name(self):
        return self.__class__.__name__
    

        with tf.Session(config=config) as sess:
    dcgan = GAN(sess, model_name=FLAGS.model_name, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  sample_size=FLAGS.sample_size, input_transform="resize", output_dir=FLAGS.output_dir, sample_interval=FLAGS.sample_interval, checkpoint_interval=FLAGS.checkpoint_interval, bbox_weight=FLAGS.bbox_weight, image_weight=FLAGS.image_weight)
    dcgan.train(FLAGS)