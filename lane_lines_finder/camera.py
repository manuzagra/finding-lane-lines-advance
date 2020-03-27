import numpy as np
import cv2
import pathlib
import pickle

from lane_lines_finder.process_step import ProcessStep


class Camera(ProcessStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matrix = None
        self.distortion_coefficients = None
        self.calibrated = False
        if (kwargs.get('matrix') and kwargs.get('distortion_coefficients')) or (kwargs.get('directory') and kwargs.get('pattern_size')):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        if kwargs.get('matrix') and kwargs.get('distortion_coefficients'):
            self.matrix = kwargs['matrix']
            self.distortion_coefficients = kwargs['distortion_coefficients']
            self.calibrated = True
        elif kwargs.get('directory') and kwargs.get('pattern_size'):
            self.calibrate(kwargs['directory'], kwargs['pattern_size'])

    def process(self, img=None, **kwargs):
        return self.undistort(img), kwargs

    def dump_calibration(self, file):
        if self.calibrated:
            with open(file, 'wb') as f:
                pickle.dump({'matrix': self.matrix, 'distortion_coefficients': self.distortion_coefficients}, f)
                return True
        else:
            print('Camera no calibrated. Calibration dump canceled.')
        return False

    def load_calibration(self, file):
        try:
            with open(file, 'rb') as f:
                cal = pickle.load(f)
                self.matrix = cal['matrix']
                self.distortion_coefficients = cal['distortion_coefficients']
                self.calibrated = True
                print(f'Camera calibrated from file "{file}".')
                return True
        except FileNotFoundError:
            print(f'File "{file}" not found.')
            return False
        except KeyError:
            print(f'File "{file}" is not a saved calibration.')
            return False

    def calibrate_from_chessboard(self, directory, pattern_size):
        """"""
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) defining the points detected in the chessboard
        obj_points_prototype = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        obj_points_prototype[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d points in real world space
        img_points = []  # 2d points in image plane.

        # All the images for calibration must have the same size
        img_size = None

        # Count of the number of images
        n_img = 0

        # Step through the list and search for chessboard corners
        for idx, file in enumerate(pathlib.Path(directory).iterdir()):
            # Get the absolute path
            file = str(file.resolve())

            # Read the image
            img = cv2.imread(file)
            if img is None:  # the file is not an image
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points, image points
            if ret:
                # Increment the count
                n_img += 1
                # Save the size of the first image
                if not img_size:
                    img_size = (img.shape[1], img.shape[0])
                # Check all the images have the same size, raise an exception if they are not
                elif abs(img_size[0] - img.shape[1]) > 10 or abs(img_size[1] - img.shape[0]) > 10:
                    raise UserWarning('All the images must have the same size (height, width).')

                # refining pixel coordinates for given 2d points.
                # cv2.cornerSubPix(image, corners, window_size, zero_zone, criteria)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                # Append the objects to the calibration matrices
                obj_points.append(obj_points_prototype)
                img_points.append(corners)
        if n_img >= 5:
            # Do camera calibration given object points and image points
            _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

            self.calibrated = True
            self.matrix = mtx
            self.distortion_coefficients = dist
            return True
        else:
            print('The directory does not have enough images, 5 or more images are required.')
            return False

    def calibrate(self, directory, pattern_size):
        directory = pathlib.Path(directory).resolve()
        for p in directory.glob('*.p'):
            if self.load_calibration(str(p)):
                return True
        self.calibrate_from_chessboard(directory, pattern_size)
        self.dump_calibration(pathlib.Path(directory).joinpath('calibration.p').resolve())
        return self.calibrated

    def undistort(self, img):
        return cv2.undistort(img, self.matrix, self.distortion_coefficients)  #, None, mtx)


def self_driving_car_camera():
    camera = Camera()
    camera.calibrate('../camera_cal', (9, 6))
    return camera


if __name__ == '__main__':
    cam = Camera()
    cam.calibrate('./test', (9, 6))

    i = 0
