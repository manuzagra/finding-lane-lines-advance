import numpy as np
import cv2
import pathlib
import pickle
from lane_lines_finder.utils import get_image, save_image

from lane_lines_finder.process_step import ProcessStep


class PerspectiveTransform(ProcessStep):
    def __init__(self, **kwargs):
        ProcessStep.__init__(self)
        self.source_points = None
        self.destination_points = None
        self.matrix = None
        self.matrix_inverse = None
        if kwargs.get('source_points') is not None and kwargs.get('destination_points') is not None:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.source_points = kwargs['source_points']
        self.destination_points = kwargs['destination_points']
        self.matrix = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        self.matrix_inverse = cv2.getPerspectiveTransform(self.destination_points, self.source_points)

    def process(self, img, **kwargs):
        if kwargs.get('inverse'):
            return self.undo_transform(img)
        else:
            return self.transform(img)

    def transform(self, img):
        return cv2.warpPerspective(img, self.matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def undo_transform(self, img):
        return cv2.warpPerspective(img, self.matrix_inverse, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


def self_driving_car_transform():
    # src_points = np.int32([[193, 720], [610, 440], [670, 440], [1120, 720]]) # to the bottom
    src_points = np.float32([[270, 670], [610, 440], [670, 440], [1040, 670]])
    dst_points = np.float32([[270, 670], [270, 50], [1040, 50], [1040, 670]])
    p = PerspectiveTransform(source_points=src_points, destination_points=dst_points)
    return p


def plots_perspective_transform():
    src_points = np.int32([[270, 670], [610, 440], [670, 440], [1040, 670]])
    for path in pathlib.Path('../test_images').glob('*straight_lines*'):
        img = get_image(path.resolve())
        for i in range(src_points.shape[0]-1):
            cv2.line(img, (src_points[i,0],src_points[i,1]), (src_points[i+1,0],src_points[i+1,1]), (0,0,255), 2)
        cv2.line(img, (src_points[0,0],src_points[0,1]), (src_points[-1,0],src_points[-1,1]), (0,0,255), 2)
        save_image(img, 'perspective_'+path.name, '../writeup_images')


if __name__ == '__main__':
    plots_perspective_transform()
    self_driving_car_transform()
