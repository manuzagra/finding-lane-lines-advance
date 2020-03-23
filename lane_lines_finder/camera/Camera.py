import numpy as np
import cv2
import pathlib
import pickle


class Camera:
    def __init__(self):
        self.matrix = None
        self.distortion_coefficients = None

    def dump_calibration(self, file):
        with open(file, 'wb') as f:
            pickle.dump({'matrix': self.matrix, 'distortion_coefficients': self.distortion_coefficients}, f)

    def load_calibration(self, file):
        with open(file) as f:
            cal = pickle.load(f)
            self.matrix = cal['matrix']
            self.distortion_coefficients = cal['distortion_coefficients']

    def calibrate(self, directory, pattern_size):
        """"""
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) defining the points detected in the chessboard
        obj_points_prototype = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        obj_points_prototype[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        obj_points = []  # 3d points in real world space
        img_points = []  # 2d points in image plane.

        # All the images for calibration must have the same size
        img_size = None

        # Step through the list and search for chessboard corners
        for idx, file in enumerate(pathlib.Path(directory).glob('*')):
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

        # Do camera calibration given object points and image points
        _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

        self.matrix = mtx
        self.distortion_coefficients = dist

    def undistort(self, img):
        return cv2.undistort(img, self.matrix, self.distortion_coefficients)  #, None, mtx)


if __name__ == '__main__':
    cam = Camera()
    cam.calibrate('./test', (9,6))

    i = 0