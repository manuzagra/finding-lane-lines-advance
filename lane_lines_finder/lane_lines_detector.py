import numpy as np
import cv2

from lane_lines_finder.process_step import ProcessStep


class Window:
    def __init__(self, width=None, heigth=None):
        self._x_range = [0, 0]
        self._y_range = [0, 0]
        if width and heigth:
            self.set_size(width, heigth)

    def set_size(self, width=None, heigth=None):
        if width is not None:
            x_mid = (self._x_range[0] + self._x_range[1]) / 2
            self._x_range = [x_mid - width / 2, x_mid + width / 2]
        if heigth is not None:
            y_mid = (self._y_range[0] + self._y_range[1]) / 2
            self._y_range = [y_mid-heigth/2, y_mid+heigth/2]

    def get_size(self):
        width = self._x_range[1] - self._x_range[0]
        heigth = self._y_range[1] - self._y_range[0]
        return width, heigth

    def set_center(self, x=None, y=None):
        width, heigth = self.get_size()
        if x is not None:
            self._x_range = [x - width / 2, x + width / 2]
        if y is not None:
            self._y_range = [y-heigth/2, y+heigth/2]

    def get_center(self, p):
        x_mid = (self._x_range[0] + self._x_range[1]) / 2
        y_mid = (self._y_range[0] + self._y_range[1]) / 2
        return x_mid, y_mid

    def set_base_center(self, x=None, y=None):
        width, heigth = self.get_size()
        if x is not None:
            self._x_range = [x - width / 2, x + width / 2]
        if y is not None:
            # the base has bigger y coordinate
            self._y_range = [y-heigth, y]

    def move(self, x=None, y=None):
        if x is not None:
            self._x_range = [c+x for c in self._x_range]
        if y is not None:
            self._y_range = [c+y for c in self._y_range]

    def get_rectangle(self):
        return (self._x_range[0], self._y_range[0]), (self._x_range[1], self._y_range[1])

    # TODO check this function
    def inner_nonzero_pixels(self, img):
        """
        :param img:
        :return: np.array with the coordinates of the white points in the window
        """
        # Identify the x and y positions of all non_zero pixels in the image
        non_zero = img.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])
        valid_non_zero_ind = ((non_zero_x >= self._x_range[0]) & (non_zero_x < self._x_range[1]) &
                          (non_zero_y >= self._y_range[0]) & (non_zero_y < self._y_range[1])).nonzero()[0]
        return list(np.vstack((non_zero_x[valid_non_zero_ind], non_zero_y[valid_non_zero_ind])).T)


class Line:
    def __init__(self, size):
        self.size = size
        # number of valid lines
        self.valid_lines = 0
        # was the line detected
        self.detected = np.array([False]*size)
        # polynomial coefficients
        self.poly = np.array([np.array([0,0,0])]*size)

    def last_detected(self):
        d = self.detected[0] if self.valid_lines else False
        return d

    def add(self, poly, detected):
        self.valid_lines = self.valid_lines + 1 if self.valid_lines < self.size else self.size
        self.detected = np.hstack([detected, self.detected[:-1]])
        self.poly = np.vstack([poly, self.poly[:-1]])

    def best(self):
        t = 0
        poly = np.zeros_like(self.poly[0])
        for i in range(self.valid_lines):
            forget_parameter = self.size - i
            if self.detected[i]:
                poly += forget_parameter * self.poly[i]
                t += forget_parameter
        poly = poly / t
        # poly = np.mean(self.poly[:self.valid_lines][self.detected[:self.valid_lines]], axis=0)
        return poly

    @classmethod
    def combine(cls, l1, l2):
        p1 = l1.best()
        p2 = l2.best()

        p = np.mean([p1, p2], axis=0)

        x1_b = p1[0] * 720**2 + p1[1] * 720 + p1[2]
        x2_b = p2[0] * 720**2 + p2[1] * 720 + p2[2]
        x_b = p[0] * 720**2 + p[1] * 720 + p[2]

        return np.array([p[0], p[1], p[2]-(x_b-x1_b)]), np.array([p[0], p[1], p[2]+(x2_b-x_b)])


class FindLines(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # parameters of the sliding window
        self.window_number = 0  # 9 number of sliding windows
        self.window_width = 0  # 100 width of the windows +/- margin
        self.window_min_n_pixels = 0  # 50 minimum number of pixels found to recenter window
        # parameters to look for a line arround a previous line
        self.search_width = 0
        # conversion between pixels and meters
        self.pixels_to_meters = (0, 0)
        # store the last lines
        self.history_len = 0
        self.lines = None
        if 'window_number' in kwargs and 'window_width' in kwargs and 'window_min_n_pixels' in kwargs \
                and 'search_width' in kwargs and 'pixels_to_meters' in kwargs and 'history_len' in kwargs:
            self.setup(**kwargs)

    def setup(self, **kwargs):
        self.window_number = kwargs['window_number']
        self.window_width = kwargs['window_width']
        self.window_min_n_pixels = kwargs['window_min_n_pixels']
        self.search_width = kwargs['search_width']
        self.pixels_to_meters = kwargs['pixels_to_meters']
        self.history_len = kwargs['history_len']
        self.lines = (Line(self.history_len), Line(self.history_len))
        self.ready()


    @classmethod
    def from_params(cls, window_number, window_width, window_min_n_pixels, search_width, pixels_to_meters, history_len):
        return cls(window_number=window_number, window_margin=window_width, window_min_n_pixels=window_min_n_pixels,
                   pixels_to_meters=pixels_to_meters, history_len=history_len)

    def process(self, img=None, **kwargs):
        self.check_ready()
        # it is going to process connected images(video) or disconnected images(image)
        if kwargs['type'] == 'image':
            polys, line_points = self.first_detection(img)
        else:  # if kwargs['type'] == 'video':
            if not (self.lines[0].last_detected() and self.lines[1].last_detected()):
                polys, line_points = self.first_detection(img)
            else:
                polys, line_points = self.detection(img)
            # security checks
            detected = self.check_lines(polys)
            # add them to the lines
            self.lines[0].add(polys[0], detected)
            self.lines[1].add(polys[1], detected)
            polys = (self.lines[0].best(), self.lines[1].best())
            # polys = Line.combine(self.lines[0], self.lines[1])

        # the offset to the centre of the road in meters
        offset_x = self.measure_offset_x(img, polys[0], polys[1])
        # Curvature of the road in m
        # we suppose lines are parallels
        av_poly = np.mean([polys[0], polys[1]], axis=0)
        # calculate the curvature on the bottom of the image
        radius = self.measure_curvature(av_poly, img.shape[0]-1)

        # save the output for the pipeline
        kwargs['lines_polynomial'] = [polys[0], polys[1]]
        kwargs['lines_points'] = [line_points[0], line_points[1]]
        kwargs['offset_x'] = offset_x
        kwargs['curvature_radius'] = radius
        return img, kwargs

    def check_lines(self, polys):
        return True  # TODO change this
        # check curvature is similar
        c_l = self.measure_curvature(polys[0], 719)
        c_r = self.measure_curvature(polys[1], 719)
        if np.absolute(c_l - c_r) > 0.5 * (c_l + c_r)/2:
            # lines do not have similar curvature
            return False

        # check distance between lines and parallelisms
        ys = [300, 500, 700]
        dist = []
        for y in ys:
            x_l = polys[0][0] * y**2 + polys[0][1] * y + polys[0][2]
            x_r = polys[1][0] * y**2 + polys[1][1] * y + polys[1][2]
            diff_m = np.absolute(x_r - x_l) * self.pixels_to_meters[0]
            if np.absolute(diff_m - 3.7) > 0.5:
                # Lines are not separated more or less 3.7 m
                return False
            dist.append(diff_m)
        if np.any(np.absolute(np.array(dist)-np.mean(dist)) > 0.5*np.mean(dist)):
            # distances in the three points arent similar
            return False
        return True

    def detection(self, img):
        # Grab activated pixels
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # generate the points in the lines
        poly_l = self.lines[0].best()
        poly_r = self.lines[1].best()
        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        x_l = poly_l[0] * y**2 + poly_l[1] * y + poly_l[2]
        x_r = poly_r[0] * y**2 + poly_r[1] * y + poly_r[2]

        # retrive the last poly
        left_fit = self.lines[0].best()
        right_fit = self.lines[1].best()
        # for every nonzero y calculate the correspond x in the polinomial function,
        # add the margin and the if the correspondent x in the nonzero x is within the margins
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - self.search_width))
                          & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + self.search_width)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - self.search_width))
                           & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + self.search_width)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        line_l_points = np.array([nonzerox[left_lane_inds], nonzeroy[left_lane_inds]]).T
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        line_r_points = np.array([nonzerox[right_lane_inds], nonzeroy[right_lane_inds]]).T

        # Fit new polynomials
        poly_l = self.fit_poly(line_l_points)
        poly_r = self.fit_poly(line_r_points)

        return (poly_l, poly_r), (line_l_points, line_r_points)

    def first_detection(self, img):
        line_l_points, line_r_points = self.sliding_window(img)
        if line_r_points.size and line_l_points.size:
            poly_l = self.fit_poly(line_l_points)
            poly_r = self.fit_poly(line_r_points)
            return (poly_l, poly_r), (line_l_points, line_r_points)
        else:

            return (np.array([0,0,0]), np.array([0,0,0])), (line_l_points, line_r_points)

    def sliding_window(self, img):
        win_l = Window(width=self.window_width, heigth=img.shape[0]/self.window_number)
        win_r = Window(width=self.window_width, heigth=img.shape[0]/self.window_number)

        # Set initial position
        win_l_x_centre, win_r_x_centre = self.find_hist_bottom_peaks(img)
        win_l.set_base_center(win_l_x_centre, img.shape[0]-1)
        win_r.set_base_center(win_r_x_centre, img.shape[0]-1)

        # To save the points owned by each line
        line_l_points = []
        line_r_points = []

        for i in range(self.window_number):
            # Get valid points within the windows
            l_points = win_l.inner_nonzero_pixels(img)
            r_points = win_r.inner_nonzero_pixels(img)

            # Save those points
            line_l_points += l_points
            line_r_points += r_points

            # If there are more points than the minimum, modify the position of the window
            if len(l_points) >= self.window_min_n_pixels:
                new_x = np.int(np.mean(np.array(l_points)[:,0]))
                win_l.set_base_center(x=new_x)
            if len(r_points) >= self.window_min_n_pixels:
                new_x = np.int(np.mean(np.array(r_points)[:,0]))
                win_r.set_base_center(x=new_x)
            win_l.move(y=-win_l.get_size()[1])
            win_r.move(y=-win_r.get_size()[1])

        return np.array(line_l_points), np.array(line_r_points)

    def fit_poly(self, points):
        # generate a second order polynomial to fit the points
        # It is a polynomial of the form x = a*y**2 + b*y + c
        # x is used a dependent variable because almost vertical lines are expected
        return np.polyfit(points[:,1], points[:,0], 2)

    def measure_curvature(self, poly, y):
        """
        Calculates the curvature of polynomial functions in meters.
        :param poly:
        :param y:
        :return:
        """
        a = (self.pixels_to_meters[0]/self.pixels_to_meters[1]**2)*poly[0]
        b = (self.pixels_to_meters[0]/self.pixels_to_meters[1])*poly[1]
        radius = ((1 + (2 * a * y + b) ** 2) ** 1.5) / np.absolute(2 * a)
        return radius

    def measure_offset_x(self, img, poly_l, poly_r):
        """
        Return the offset of the car in the x axis referenced to the center of the road in meter
        :param img:
        :param poly_l:
        :param poly_r:
        :return:
        """
        y = img.shape[0]    # we are going to calculate the x value on the bottom of the image
        x_l = poly_l[0] * y**2 + poly_l[1] * y + poly_l[2]
        x_r = poly_r[0] * y**2 + poly_r[1] * y + poly_r[2]
        middle_road = (x_l + x_r) / 2
        diff = np.absolute(middle_road - img.shape[1]/2)
        return self.pixels_to_meters[0] * diff

    def find_hist_bottom_peaks(self, img):
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0] // 2:, :]

        # Sum across image pixels vertically - make sure to set an `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base




