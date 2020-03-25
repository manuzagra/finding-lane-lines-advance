import numpy as np


class Pipeline:
    def __init__(self, steps=[]):
        self.steps = steps

    def append(self, step):
        self.steps.append(step)

    def process(self, in_img, **kwargs):
        img = np.copy(in_img)

        for step in self.steps:
            img = step.process(img, **kwargs)

        return img


def self_driving_car_pipeline():
    from lane_lines_finder.utils import get_image, save_image
    import lane_lines_finder.steps as step
    import lane_lines_finder.camera as cam

    p = Pipeline()

    p.append(cam.self_driving_car_camera())





