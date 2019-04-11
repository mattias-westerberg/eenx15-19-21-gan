from abc import ABC, abstractmethod

class Disctriminator:
    def __init__(self):
        super(Disctriminator, self).__init__()

    @abstractmethod
    def __call__(self, image, reuse=False, is_training=False):
        pass
        
    def name(self):
        return self.__class__.__name__