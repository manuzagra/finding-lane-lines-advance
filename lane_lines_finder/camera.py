import numpy as np
import cv2
import pathlib
import pickle

from lane_lines_finder.process_step import ProcessStep


class Camera(ProcessStep):
    """
    It encapsulate the parameters of the camera and correct the distortion produced by the lens
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.matrix = None
        self.distortion_coefficients = None
        self.calibrated = False
        if (kwargs.get('matrix') and kwargs.get('distortion_coefficients')) or (kwargs.get('directory') and kwargs.get('pattern_size')):
            self.setup(**kwargs)

    def setup(self, **kwargs):
        # It is possible to set it up directly with the parameters or computating the calibration
        if kwargs.get('matrix') and kwargs.get('distortion_coefficients'):
            self.matrix = kwargs['matrix']
            self.distortion_coefficients = kwargs['distortion_coefficients']
            self.calibrated = True
        elif kwargs.get('directory') and kwargs.get('pattern_size'):
            self.calibrate(kwargs['directory'], kwargs['pattern_size'])

    def process(self, img=None, **kwargs):
        # Undistort the image
        return self.undistort(img), kwargs

    def dump_calibration(self, file):
        """
        Dump the calibration parameters
        :param file: path where to dump the parameters
        :return: bool, depending if the dump could be done
        """
        if self.calibrated:
            with open(file, 'wb') as f:
                pickle.dump({'matrix': self.matrix, 'distortion_coefficients': self.distortion_coefficients}, f)
                return True
        else:
            print('Camera no calibrated. Calibration dump canceled.')
        return False

    def load_calibration(self, file):
        """
        Load the calibration parameters from a file.
        :param file: path to the file with the dumped configuration parameters
        :return: True if the file was found and it contains the parameters, False otherwise
        """
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
        """
        Calibrate the camera parameters using chessboard patterns.
        :param directory: path to the directory containing the images of the chessboard
        :param pattern_size: number of vertices inside the chessboard
        :return: True if the camera is successfully calibrated, False otherwise
        """
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
        # Around 20 images is a good number, less than 5 the calibration can not be good
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
        """
        General method to calibrate a camera.
        It will look for dumped parameters and calibrate from chessboard if the dumped file is not found.
        :param directory: directory to look for everything
        :param pattern_size: to use if a dump file is not found
        :return: True if the camera have been calibrated, False otherwise
        """
        directory = pathlib.Path(directory).resolve()
        # look for dumped file in the directory and load it if it exist
        for p in directory.glob('*.p'):
            if self.load_calibration(str(p)):
                return True
        # calibrate from chessboard
        self.calibrate_from_chessboard(directory, pattern_size)
        # save the calibration parameters
        self.dump_calibration(pathlib.Path(directory).joinpath('calibration.p').resolve())
        return self.calibrated

    def undistort(self, img):
        # undo the distortion
        return cv2.undistort(img, self.matrix, self.distortion_coefficients)  #, None, mtx)


if __name__ == '__main__':
    from lane_lines_finder.self_driving_car import camera
    from lane_lines_finder.utils import get_image, save_image

    c = camera()

    img = get_image('../camera_cal/calibration1.jpg')
    p_img = c.undistort(img)
    save_image(img, '../writeup_images/camera_calibration_before.jpg')
    save_image(p_img, '../writeup_images/camera_calibration_after.jpg')

