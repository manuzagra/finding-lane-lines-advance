import cv2

from lane_lines_finder.process_step import ProcessStep


class PerspectiveTransform(ProcessStep):
    """
    Given a set of input points and output points it calculates the correspondent transform
    and it is able to produce that transform in images.
    """
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
        # calculate the matrix and the inverse matrix of the transform
        self.matrix = cv2.getPerspectiveTransform(kwargs['source_points'], kwargs['destination_points'])
        self.matrix_inverse = cv2.getPerspectiveTransform(kwargs['destination_points'], kwargs['source_points'])
        # True to undo the transform
        self.inverse = kwargs['inverse']

    @classmethod
    def from_params(cls, source_points, destination_points, inverse=False):
        return cls(source_points=source_points, destination_points=destination_points, inverse=inverse)

    def process(self, img=None, **kwargs):
        if self.inverse:
            return self.undo_transform(img), kwargs
        else:
            return self.transform(img), kwargs

    def transform(self, img):
        return cv2.warpPerspective(img, self.matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def undo_transform(self, img):
        return cv2.warpPerspective(img, self.matrix_inverse, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    import pathlib
    import lane_lines_finder.utils as utils
    import lane_lines_finder.self_driving_car as self_driving_car


    def plots_perspective_transform_lines():
        """
        Function to get images of the process
        :return:
        """
        perspective = self_driving_car.perspective_transform()
        points = self_driving_car.transform_points()
        for path in pathlib.Path('../test_images').glob('*straight_lines*'):
            img = utils.get_image(path.resolve())
            img_transform = perspective.transform(img)
            for i in range(-1, points[0].shape[0] - 1):
                cv2.line(img, (points[0][i, 0], points[0][i, 1]), (points[0][i + 1, 0], points[0][i + 1, 1]),
                         (0, 0, 255), 2)
            utils.save_image(img, 'perspective_source_' + path.name, '../writeup_images')
            for i in range(-1, points[1].shape[0] - 1):
                cv2.line(img_transform, (points[1][i, 0], points[1][i, 1]), (points[1][i + 1, 0], points[1][i + 1, 1]),
                         (0, 0, 255), 2)
            utils.save_image(img_transform, 'perspective_destination_' + path.name, '../writeup_images')

    plots_perspective_transform_lines()
