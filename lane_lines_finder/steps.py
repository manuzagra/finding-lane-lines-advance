import numpy as np
import cv2
import pathlib
from lane_lines_finder.process_step import ProcessStep
from lane_lines_finder.camera import Camera
from lane_lines_finder.perspective_transform import PerspectiveTransform
from lane_lines_finder.lane_lines_detector import FindLines


class DrawText(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.function = None
        self.color = None
        self.coordinates = None
        if 'function' in kwargs and 'color' in kwargs and 'coordinates' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.function = kwargs['function']
        self.color = kwargs['color']
        self.coordinates = kwargs['coordinates']
        self.ready()

    @classmethod
    def from_params(cls, function, color, coordinates):
        return cls(function=function, color=color, coordinates=coordinates)

    def process(self, img=None, **kwargs):
        self.check_ready()
        text = self.function(img, **kwargs)
        cv2.putText(img, text, self.coordinates, cv2.FONT_HERSHEY_DUPLEX, 0.8, self.color)
        return img, kwargs


class DrawPolygon(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.points = None
        self.color = None
        self.thickness = None
        if 'points' in kwargs and 'color' in kwargs and 'thickness' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.points = kwargs['points']
        self.color = kwargs['color']
        self.thickness = kwargs['thickness']
        self.ready()

    @classmethod
    def from_params(cls, points, color, thickness):
        return cls(points=points, color=color, thickness=thickness)

    def process(self, img=None, **kwargs):
        self.check_ready()
        for i in range(-1, self.points.shape[0]-1):
            cv2.line(img, (self.points[i,0],self.points[i,1]), (self.points[i+1,0],self.points[i+1,1]), self.color, self.thickness)
        return img, kwargs


class FillPolygon(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.function = None
        self.color = None
        if 'function' in kwargs and 'color' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.function = kwargs['function']
        self.color = kwargs['color']
        self.ready()

    @classmethod
    def from_params(cls, function, color):
        return cls(function=function, color=color)

    def process(self, img=None, **kwargs):
        self.check_ready()
        points = self.function(img, **kwargs)
        cv2.fillPoly(img, np.int_([points]), self.color)
        #cv2.fillConvexPoly(img, points, self.color)
        return img, kwargs


class ColorPoints(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.function = None  # it returns the values of a cv2.fitPoly()
        self.color = None
        if 'function' in kwargs and 'color' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.function = kwargs['function']
        self.color = kwargs['color']
        self.ready()

    @classmethod
    def from_params(cls, function, color):
        return cls(function=function, color=color)

    def process(self, img=None, **kwargs):
        self.check_ready()
        points = self.function(img, **kwargs)
        img[points[:,1], points[:,0]] = list(self.color)
        return img, kwargs


class DrawPolynomial(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.function = None  # it returns the values of a cv2.fitPoly()
        self.color = None
        if 'function' in kwargs and 'color' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.function = kwargs['function']
        self.color = kwargs['color']
        self.ready()

    @classmethod
    def from_params(cls, function, color):
        return cls(function=function, color=color)

    def process(self, img=None, **kwargs):
        self.check_ready()
        poly = self.function(img, **kwargs)
        y = np.linspace(0, img.shape[0] - 1, img.shape[0]).astype(np.int)
        x = (poly[0] * y ** 2 + poly[1] * y + poly[2]).astype(np.int)
        img[y, x] = list(self.color)
        return img, kwargs


class ClearImage(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        self.ready()

    @classmethod
    def from_params(cls):
        return cls()

    def process(self, img=None, **kwargs):
        self.check_ready()
        return np.zeros_like(img), kwargs


class SelectChannel(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channel = None
        if 'channel' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.channel = kwargs['channel']
        self.ready()

    @classmethod
    def from_params(cls, channel):
        return cls(channel=channel)

    def process(self, img=None, **kwargs):
        self.check_ready()
        return img[:, :, self.channel], kwargs


class Binary2Color(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (255, 255, 255)
        if 'color' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.color = kwargs['color']
        self.ready()

    @classmethod
    def from_params(cls, color=(255, 255, 255)):
        return cls(color=color)

    def process(self, img=None, **kwargs):
        self.check_ready()
        return np.dstack((img*self.color[0], img*self.color[1], img*self.color[2])), kwargs


class CombineBinary(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipelines = None
        self.function = None  # combine the output of the different pipelines
        if 'function' in kwargs and 'pipelines' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.pipelines = kwargs['pipelines']
        self.function = kwargs['function']
        self.ready()

    @classmethod
    def from_params(cls, combining_function):
        return cls(function=combining_function)

    def process(self, img=None, **kwargs):
        self.check_ready()
        images = []
        for pip in self.pipelines:
            # TODO consider merging conflicts between kwargs from different pipelines
            image, kwargs = pip.process(img, **kwargs)
            images.append(image)
        combined = np.zeros_like(img[:,:,0])
        combined[self.function(images, **kwargs)] = 1
        return combined, kwargs


class CombineImages(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipelines = None
        self.weights = None
        if 'pipelines' in kwargs and 'weights'in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.pipelines = kwargs['pipelines']
        self.weights = kwargs['weights']
        self.ready()

    @classmethod
    def from_params(cls, pipelines, weights):
        return cls(pipelines=pipelines, weights=weights)

    def process(self, img=None, **kwargs):
        self.check_ready()
        combined = np.zeros_like(img)
        for pip, weight in zip(self.pipelines, self.weights):
            image, kwargs = pip.process(img, **kwargs)
            # cv2.addWeighted(combined, α, image, β, γ) -> combined * α + image * β + γ
            combined = cv2.addWeighted(combined, 1., image, weight, 0.)
        return combined, kwargs


class SaveImage(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.directory = kwargs.get('directory', '')
        self.prefix = kwargs.get('prefix', '')
        self.postfix = kwargs.get('postfix', '')
        if 'directory' in kwargs and 'prefix' in kwargs and 'postfix' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.directory = kwargs['directory']
        self.prefix = kwargs['prefix']
        self.postfix = kwargs['postfix']
        self.ready()

    @classmethod
    def from_params(cls, directory='', prefix='', postfix=''):
        return cls(directory=directory, prefix=prefix, postfix=postfix)

    def process(self, img=None, **kwargs):
        self.check_ready()
        name = self.prefix + kwargs['file'].split('.')[0] + self.postfix + '.' + kwargs['file'].split('.')[1]
        if self.directory:
            file = pathlib.Path(self.directory).joinpath(name).resolve()
        else:
            file = pathlib.Path(kwargs['output_directory']).joinpath(name).resolve()
        cv2.imwrite(str(file), img)
        return img, kwargs


class GaussianBlur(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = 5
        if 'kernel_size' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.kernel_size = kwargs['kernel_size']
        self.ready()

    @classmethod
    def from_params(cls, kernel_size):
        return cls(kernel_size=kernel_size)

    def process(self, img=None, **kwargs):
        self.check_ready()
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0), kwargs


# TODO delete this one, use ColorConvert
class Grayscale(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flag = cv2.COLOR_BGR2GRAY
        if 'flag' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.flag = kwargs['flag']
        self.ready()

    @classmethod
    def from_params(cls, flag=cv2.COLOR_BGR2GRAY):
        return cls(flag=flag)

    def process(self, img=None, **kwargs):
        self.check_ready()
        return cv2.cvtColor(img, self.flag), kwargs


class ConvertColor(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flag = None
        if 'flag' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.flag = kwargs['flag']
        self.ready()

    @classmethod
    def from_params(cls, flag):
        return cls(flag=flag)

    def process(self, img=None, **kwargs):
        self.check_ready()
        return cv2.cvtColor(img, self.flag), kwargs


class HistogramEqualization(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        self.ready()

    @classmethod
    def from_params(cls):
        return cls()

    def process(self, img=None, **kwargs):
        self.check_ready()
        image = np.copy(img)
        if len(img.shape) < 3:
            image[:, :] = cv2.equalizeHist(img[:, :])
        else:
            for channel in img.shape[2]:
                image[:, :, channel] = cv2.equalizeHist(img[:, :, channel])
        return image, kwargs


class Threshold(ProcessStep):
    """
    :param img: 2D np.array with values between 0 and 255
    :param thresholds: (min, max)
    :return: 2D binary np.array with the same shape than the input, with 1s where the pixel is between the thresholds
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.min = None
        self.max = None
        if 'min' in kwargs and 'max' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.min = kwargs['min']
        self.max = kwargs['max']
        self.ready()

    @classmethod
    def from_params(cls, min_thres, max_thres):
        return cls(min=min_thres, max=max_thres)

    def process(self, img=None, **kwargs):
        self.check_ready()
        out = np.zeros_like(img)
        out[(img >= self.min) & (img <= self.max)] = 1
        return out, kwargs

# TODO put all the gradient stuff in one step with a selection parameter


class AbsSobel(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.axis = None
        self.kernel_size = 0
        if 'axis' in kwargs and 'kernel_size' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        if kwargs['axis'] not in ['x', 'y']:
            raise ValueError('Axis for Sobel must be "x" or "y".')
        self.axis = (1, 0) if kwargs['axis'] == 'x' else (0, 1)
        self.kernel_size = kwargs['kernel_size']
        self.ready()

    @classmethod
    def from_params(cls, axis, kernel_size):
        return cls(axis=axis, kernel_size=kernel_size)

    def process(self, img=None, **kwargs):
        self.check_ready()
        sobel = cv2.Sobel(img, cv2.CV_64F, self.axis[0], self.axis[1], ksize=self.kernel_size)
        abs_sobel = np.absolute(sobel)
        abs_sobel_8u = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        return abs_sobel_8u, kwargs


class GradienteMag(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = 3
        if 'kernel_size' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.kernel_size = kwargs['kernel_size']
        self.ready()

    @classmethod
    def from_params(cls, kernel_size):
        return cls(kernel_size=kernel_size)

    def process(self, img=None, **kwargs):
        self.check_ready()
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        gradmag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        gradmag_8u = np.uint8(255 * gradmag / np.max(gradmag))
        return gradmag_8u, kwargs


class GradientDir(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = 3
        if 'kernel_size' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.kernel_size = kwargs['kernel_size']
        self.ready()

    @classmethod
    def from_params(cls, kernel_size):
        return cls(kernel_size=kernel_size)

    def process(self, img=None, **kwargs):
        self.check_ready()
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
        absgraddir_8u = np.uint8(255 * absgraddir / np.max(absgraddir))
        return absgraddir_8u, kwargs

