import numpy as np


class Pipeline:
    """
    Represents a pipeline.
    It is a composition of objects from classes that derive from ProcessStep.
    """
    def __init__(self):
        self.steps = []

    def append(self, s):
        """
        Append a step at the end of the pipeline
        :param s: ProcessStep derived class
        :return:
        """
        self.steps.append(s)

    def process(self, in_img, **kwargs):
        """
        Process an image performing all the steps in the pipeline
        :param in_img: image
        :param kwargs: other parameters needed for the pipeline
        :return: the processed image and additional information that may be needed for another pipeline
        """
        img = np.copy(in_img)
        for s in self.steps:
            img, kwargs = s.process(img, **kwargs)
        return img, kwargs

    def __add__(self, other):
        """
        Overload of the "+" operator. You can concatenate diferent pipelines just using "+"
        :param other: Pipeline
        :return: Pipeline with the composition of the two pipelines
        """
        if isinstance(other, Pipeline):
            p = Pipeline()
            p.steps = self.steps + other.steps
            return p
        else:
            return other.__radd__(self)
