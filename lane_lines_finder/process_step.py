from abc import ABC, abstractmethod
from lane_lines_finder.pipeline import Pipeline


class ProcessStep(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()
        self._ready = False

    @abstractmethod
    def setup(self, **kwargs):
        pass

    def check_ready(self):
        if not self._ready:
            raise RuntimeError('First you have to set up the parameters, then you can process images.')

    def ready(self):
        self._ready = True

    @abstractmethod
    def process(self, img=None, **kwargs):
        pass

    def __add__(self, other):
        p = Pipeline()
        if isinstance(other, ProcessStep):
            p.append(self)
            p.append(other)
        elif isinstance(other, Pipeline):
            p.steps = [self] + other.steps
        else:
            raise NotImplemented()
        return p

    def __radd__(self, other):
        p = Pipeline()
        if isinstance(other, Pipeline):
            p.steps = other.steps +[self]
        else:
            raise NotImplemented()
        return p
