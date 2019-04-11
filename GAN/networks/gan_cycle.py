from .gan import GAN
from .models.g_even import EvenGenerator
from .models.d_test import TestDisctriminator

class CycleGAN(GAN):
    def __init__(self):
        GAN.__init__(
            self,
            "CycleGAN",
            EvenGenerator(256),
            TestDisctriminator(256),
            image_size=256,
            c_dim=3,
            output_dir="outputs",
            bbox_weight=1.0,
            image_weight=1.0)