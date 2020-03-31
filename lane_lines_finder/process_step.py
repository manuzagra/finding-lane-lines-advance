from abc import ABC, abstractmethod
from lane_lines_finder.pipeline import Pipeline


class ProcessStep(ABC):
    """
    Interface for all the steps of the pipeline.
    """
    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()
        # It is true if all the needed parameters of the step are correctly settled up
        self._ready = False

    @abstractmethod
    def setup(self, **kwargs):
        """
        This method set up all the parametes of the step.
        Kwargs must contain ALL the needed parameters.
        It must call self.ready() if it successfully set up everything
        :param kwargs:
        :return:
        """
        # TODO decorate with self.ready(), then the derived classes do not have to do it
        pass

    def check_ready(self):
        """
        Throws an exception if not all the parameters are settled up correctly.
        It must be called before to perform any processing action.
        :return:
        """
        if not self._ready:
            raise RuntimeError('First you have to set up the parameters, then you can process images.')

    def ready(self):
        """
        Set self.ready to True.
        It must be called at the end of a successful self:setup()
        :return:
        """
        self._ready = True

    @abstractmethod
    def process(self, img=None, **kwargs):
        """
        This method will be called on every image/frame.
        It performs the different actions in the image.
        :param img:
        :param kwargs:
        :return:
        """
        pass

    def __add__(self, other):
        """
        Pipeline = Step + Step
        Pipeline = Step + Pipeline
        :param other:
        :return:
        """
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
        """
        Pipeline = Pipeline + Step
        :param other:
        :return:
        """
        p = Pipeline()
        if isinstance(other, Pipeline):
            p.steps = other.steps +[self]
        else:
            raise NotImplemented()
        return p
