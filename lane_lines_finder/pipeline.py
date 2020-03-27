import numpy as np
import cv2


class Pipeline:
    def __init__(self):
        self.steps = []

    def append(self, s):
        self.steps.append(s)

    def process(self, in_img, **kwargs):
        img = np.copy(in_img)
        for s in self.steps:
            img, kwargs = s.process(img, **kwargs)
        return img, kwargs

    def __add__(self, other):
        if isinstance(other, Pipeline):
            p = Pipeline()
            p.steps = self.steps + other.steps
            return p
        else:
            return other.__radd__(self)


def self_driving_car_pipeline():
    import lane_lines_finder.steps as step

    # grad_x = step.self_driving_car_camera() + step.Grayscale() + step.GaussianBlur(kernel_size=3) \
    #             + step.AbsSobel(axis='x', kernel_size=3) + step.Threshold(min=40, max=100) + step.Binary2Color(color=(0,255,0))
    # grad_y = step.self_driving_car_camera() + step.Grayscale() + step.GaussianBlur(kernel_size=7) \
    #             + step.AbsSobel(axis='y', kernel_size=3) + step.Threshold(min=150, max=255) + step.Binary2Color(color=(0,0,255))
    # grad_dir = step.self_driving_car_camera() + step.Grayscale() + step.GaussianBlur(kernel_size=5) \
    #             + step.GradientDir(kernel_size=5) + step.Threshold(min=np.pi/2, max=np.pi) + step.Binary2Color(color=(0,0,255))
    # grad_mag = step.self_driving_car_camera() + step.Grayscale() + step.GaussianBlur(kernel_size=5) \
    #             + step.GradienteMag(kernel_size=5) + step.Threshold(min=100, max=255) + step.Binary2Color(color=(0,0,255))

    # channel_s = step.self_driving_car_camera() + step.ConvertColor(flag=cv2.COLOR_BGR2HLS) + step.SelectChannel(channel=2) \
    #             + step.Threshold(min=170, max=255) + step.Binary2Color(color=(255,0,0))

    # channel_b = step.self_driving_car_camera() + step.ConvertColor(flag=cv2.COLOR_BGR2LAB) + step.SelectChannel(channel=2) \
    #             + step.Threshold(min=170, max=255) + step.Binary2Color(color=(255,0,0))
    #
    # channel_l = step.self_driving_car_camera() + step.ConvertColor(flag=cv2.COLOR_BGR2LAB) + step.SelectChannel(channel=0) \
    #             + step.HistogramEqualization() + step.Threshold(min=180, max=255) + step.Binary2Color(color=(255,0,0))

    # combined = step.CombineImages.from_params([grad_x, channel_s], [1,1])
    #             + step.DrawPolygon(points=perspective.self_driving_car_transform_points()[0], color=(0,0,255), thickness=2)
    # transformed = combined + step.self_driving_car_transform()

    grad_x = step.self_driving_car_camera() + step.Grayscale() + step.HistogramEqualization() \
             + step.GaussianBlur(kernel_size=9) + step.AbsSobel(axis='x', kernel_size=3) \
             + step.Threshold(min=50, max=255)
    channel_s = step.self_driving_car_camera() \
                + step.ConvertColor(flag=cv2.COLOR_BGR2HLS) + step.SelectChannel(channel=2) \
                + step.Threshold(min=170, max=255)
    combined = step.CombineBinary(pipelines=[grad_x, channel_s],
        function=lambda imgs: (imgs[0] == 1) | (imgs[1] == 1)) \
               + step.Binary2Color(color=(255, 255, 255))

    return step.SaveImage(postfix='_0') + combined + step.SaveImage(postfix='_1_binary') \
           + step.self_driving_car_transform() + step.SaveImage(postfix='_2_transform')


