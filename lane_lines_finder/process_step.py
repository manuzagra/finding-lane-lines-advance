from abc import ABC, abstractmethod


class ProcessStep(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        ABC.__init__(self)
        self.ready = False

    @abstractmethod
    def setup(self, **kwargs):
        pass

    def check_ready(self):
        if not self.ready:
            raise RuntimeError('First you have to set up the parameters, then you can process images.')

    def ready(self):
        self.ready = True

    @abstractmethod
    def process(self, img, **kwargs):
        pass
