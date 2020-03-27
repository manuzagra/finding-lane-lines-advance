import numpy as np
import cv2

from lane_lines_finder.process_step import ProcessStep


class PerspectiveTransform(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.source_points = None
        # self.destination_points = None
        self.matrix = None
        self.matrix_inverse = None
        self.inverse = None
        if kwargs.get('source_points') is not None and kwargs.get('destination_points') is not None:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        # self.source_points = kwargs['source_points']
        # self.destination_points = kwargs['destination_points']
        self.matrix = cv2.getPerspectiveTransform(kwargs['source_points'], kwargs['destination_points'])
        self.matrix_inverse = cv2.getPerspectiveTransform(kwargs['destination_points'], kwargs['source_points'])
        self.inverse = kwargs['inverse']

    @classmethod
    def from_params(cls, source_points, destination_points, inverse=False):
        return cls(source_points=source_points, destination_points=destination_points, inverse=inverse)

    def process(self, img=None, **kwargs):
        if kwargs.get('inverse'):
            return self.undo_transform(img), kwargs
        else:
            return self.transform(img), kwargs

    def transform(self, img):
        return cv2.warpPerspective(img, self.matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def undo_transform(self, img):
        return cv2.warpPerspective(img, self.matrix_inverse, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


def self_driving_car_transform_points():
    # src_points = np.float32([[270, 670], [610, 440], [670, 440], [1040, 670]])
    # dst_points = np.float32([[270, 670], [270, 50], [1040, 50], [1040, 670]])
    # src_points = np.float32([[193, 720], [610, 440], [670, 440], [1120, 720]])  # from to the bottom using lines
    # dst_points = np.float32([[280, 720], [280, 0], [1000, 0], [1000, 720]])  # from to the bottom using lines
    # src_points = np.float32([[193, 720], [575, 460], [710, 460], [1120, 720]])  # from middle to the bottom using lines
    # dst_points = np.float32([[280, 720], [280, 0], [1000, 0], [1000, 720]])  # from middle to the bottom using lines

    src_points = np.float32([[200, 720], [580, 460], [700, 460], [1080, 720]])  # from middle to the bottom image centered
    dst_points = np.float32([[280, 720], [280, 0], [1000, 0], [1000, 720]])  # from middle to the bottom image centered
    return src_points, dst_points


def self_driving_car_transform(inverse=False):
    return PerspectiveTransform(source_points=self_driving_car_transform_points()[0], destination_points=self_driving_car_transform_points()[1], inverse=inverse)


if __name__ == '__main__':
    import pathlib
    import lane_lines_finder.utils as utils


    def plots_perspective_transform_lines():
        perspective = self_driving_car_transform()
        points = self_driving_car_transform_points()
        for path in pathlib.Path('../test_images').glob('*straight_lines*'):
            img = utils.get_image(path.resolve())
            img_transform = perspective.transform(img)
            # utils.save_image(perspective.undo_transform(img_transform), 'perspective_ret_'+path.name, '../writeup_images')
            for i in range(-1, points[0].shape[0] - 1):
                cv2.line(img, (points[0][i, 0], points[0][i, 1]), (points[0][i + 1, 0], points[0][i + 1, 1]),
                         (0, 0, 255), 2)
            utils.save_image(img, 'perspective_source_' + path.name, '../writeup_images')
            for i in range(-1, points[1].shape[0] - 1):
                cv2.line(img_transform, (points[1][i, 0], points[1][i, 1]), (points[1][i + 1, 0], points[1][i + 1, 1]),
                         (0, 0, 255), 2)
            utils.save_image(img_transform, 'perspective_destination_' + path.name, '../writeup_images')

    plots_perspective_transform_lines()
