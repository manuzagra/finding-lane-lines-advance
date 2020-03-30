import numpy as np


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


    # return step.SaveImage.from_params(postfix='_0') + combined + step.SaveImage.from_params(postfix='_1_binary') \
    #     + perspective_transform() + step.SaveImage.from_params(postfix='_2_transform') + find_lane_lines('image') \
    #     + step.Binary2Color(color=(255,255,255)) \
    #     + step.ColorPoints(function=lambda imgs, **kwargs: kwargs['lines_points'][0], color=(255,0,0)) \
    #     + step.ColorPoints(function=lambda imgs, **kwargs: kwargs['lines_points'][1], color=(0,255,0)) \
    #     + step.SaveImage.from_params(postfix='_3_lines_color') \
    #     + step.DrawPolynomial(function=lambda imgs, **kwargs: kwargs['lines_polynomial'][0], color=(0,0,255)) \
    #     + step.DrawPolynomial(function=lambda imgs, **kwargs: kwargs['lines_polynomial'][1], color=(0,0,255)) \
    #     + step.SaveImage.from_params(postfix='_4_lines_lines')