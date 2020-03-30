import numpy as np
import cv2


def camera():
    from lane_lines_finder.camera import Camera
    cam = Camera()
    cam.calibrate('../camera_cal', (9, 6))
    return cam


def transform_points():
    # src_points = np.float32([[270, 670], [610, 440], [670, 440], [1040, 670]])
    # dst_points = np.float32([[270, 670], [270, 50], [1040, 50], [1040, 670]])
    # src_points = np.float32([[193, 720], [610, 440], [670, 440], [1120, 720]])  # from to the bottom using lines
    # dst_points = np.float32([[280, 720], [280, 0], [1000, 0], [1000, 720]])  # from to the bottom using lines
    # src_points = np.float32([[193, 720], [575, 460], [710, 460], [1120, 720]])  # from middle to the bottom using lines
    # dst_points = np.float32([[280, 720], [280, 0], [1000, 0], [1000, 720]])  # from middle to the bottom using lines

    src_points = np.float32([[200, 720], [580, 460], [700, 460], [1080, 720]])  # from middle to the bottom image centered
    dst_points = np.float32([[280, 720], [280, 0], [1000, 0], [1000, 720]])  # from middle to the bottom image centered
    return src_points, dst_points


def perspective_transform(inverse=False):
    from lane_lines_finder.perspective_transform import PerspectiveTransform
    return PerspectiveTransform(source_points=transform_points()[0], destination_points=transform_points()[1], inverse=inverse)


def pixels_to_meters():
    return 3.7/700, 30/720  # meters per pixel in x,y dimensions


def find_lane_lines(input_type='image'):
    from lane_lines_finder.lane_lines_detector import FindLines
    return FindLines(window_number=10, window_width=150, window_min_n_pixels=50, search_width=150,
                     pixels_to_meters=pixels_to_meters(), history_len=25)


def pipeline():
    import lane_lines_finder.steps as step

    grad_x = camera() + step.Grayscale.from_params() + step.HistogramEqualization() \
        + step.GaussianBlur(kernel_size=9) + step.AbsSobel(axis='x', kernel_size=3) + step.Threshold(min=50, max=255)
    channel_s = camera() \
        + step.ConvertColor(flag=cv2.COLOR_BGR2HLS) + step.SelectChannel(channel=2) + step.Threshold(min=170, max=255)
    combined = step.CombineBinary(pipelines=[grad_x, channel_s],
                                  function=lambda imgs, **kwargs: (imgs[0] == 1) | (imgs[1] == 1))

    def generate_polygon_points(img, **kwargs):
        poly_l = kwargs['lines_polynomial'][0]
        poly_r = kwargs['lines_polynomial'][1]
        y = np.linspace(0, img.shape[0] - 1, img.shape[0]).astype(np.int)
        x_l = (poly_l[0] * y ** 2 + poly_l[1] * y + poly_l[2]).astype(np.int)
        x_r = (poly_r[0] * y ** 2 + poly_r[1] * y + poly_r[2]).astype(np.int)
        return np.vstack([np.vstack([x_l, y]).T, np.flipud(np.vstack([x_r, y]).T)])

    drivable_area = combined + perspective_transform() + find_lane_lines('image') \
        + step.Binary2Color(color=(255, 255, 255)) + step.ClearImage() \
        + step.FillPolygon(function=generate_polygon_points, color=(0, 255, 0)) \
        + perspective_transform(inverse=True)

    complete = step.CombineImages.from_params(pipelines=[camera(), drivable_area], weights=[1, 0.6]) \
        + step.DrawText(function=lambda img, **kwargs: 'Offset = ' + str(int(kwargs['offset_x']*100)/100.) + '    Curvature = ' + str(int(kwargs['curvature_radius'])),
                        color=(255,0,0), coordinates=(400, 680))
    return complete


