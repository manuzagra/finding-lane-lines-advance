import numpy as np
import cv2
import pathlib
from lane_lines_finder.process_step import ProcessStep


class Binary2Color(ProcessStep):
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.color = None
        if kwargs.get('color'):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.color = kwargs['color']
        self.ready()

    def process(self, img, **kwargs):
        self.check_ready()
        return np.dstack((img*self.color[0], img*self.color[1], img*self.color[2]))


class SaveImage(ProcessStep):
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.directory = None
        self.name = None
        if kwargs.get('directory') and kwargs.get('name'):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.directory = kwargs['directory']
        self.name = kwargs['name']
        self.ready()

    def process(self, img, **kwargs):
        self.check_ready()
        file = pathlib.Path(self.directory).joinpath(self.name).resolve()
        cv2.imwrite(str(file), img)
        return img


class Grayscale(ProcessStep):
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.flag = cv2.COLOR_BGR2GRAY
        if kwargs.get('flag'):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.flag = kwargs['flag']
        self.ready()

    def process(self, img, **kwargs):
        self.check_ready()
        return cv2.cvtColor(img, self.flag)


class HLS(ProcessStep):
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.flag = cv2.COLOR_RGB2HLS
        if kwargs.get('flag'):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.flag = kwargs['flag']
        self.ready()

    def process(self, img, **kwargs):
        self.check_ready()
        return cv2.cvtColor(img, self.flag)


class Threshold(ProcessStep):
    """
    :param img: 2D np.array with values between 0 and 255
    :param thresholds: (min, max)
    :return: 2D binary np.array with the same shape than the input, with 1s where the pixel is between the thresholds
    """
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.min = kwargs.get('min', 0)
        self.max = kwargs.get('max', 0)

    def setup(self, **kwargs):
        self.min = kwargs['min']
        self.max = kwargs['max']
        self.ready()

    def process(self, img, **kwargs):
        self.check_ready()
        out = np.zeros_like(img)
        out[(img >= self.min) & (img <= self.max)] = 1
        return out


class AbsSobel(ProcessStep):
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.axis = None
        self.kernel_size = 0
        if kwargs.get('axis') and kwargs.get('kernel_size'):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        if kwargs['axis'] not in ['x', 'y']:
            raise ValueError('Axis for Sobel must be "x" or "y".')
        self.axis = (1, 0) if kwargs['axis'] == 'x' else (0, 1)
        self.kernel_size = kwargs['kernel_size']
        self.ready()

    def process(self, img, **kwargs):
        self.check_ready()
        sobel = cv2.Sobel(img, cv2.CV_64F, self.axis[0], self.axis[1], ksize=self.kernel_size)
        abs_sobel = np.absolute(sobel)
        abs_sobel_8u = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        return abs_sobel_8u


class GradienteMag(ProcessStep):
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.kernel_size = 0
        if kwargs.get('kernel_size'):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.kernel_size = kwargs['kernel_size']
        self.ready()

    def process(self, img, **kwargs):
        self.check_ready()
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        gradmag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        gradmag_8u = np.uint8(255 * gradmag / np.max(gradmag))
        return gradmag_8u


class GradientDir(ProcessStep):
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.kernel_size = 0
        if kwargs.get('kernel_size'):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.kernel_size = kwargs['kernel_size']
        self.ready()

    def process(self, img, **kwargs):
        self.check_ready()
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
        return absgraddir

