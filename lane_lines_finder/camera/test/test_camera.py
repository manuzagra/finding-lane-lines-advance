import unittest
from lane_lines_finder.camera.Camera import Camera
import pathlib
import os


class TestCamera(unittest.TestCase):

    def test_calibrate(self):
        directory = pathlib.Path('.')
        cam = Camera()
        cam.calibrate(str(directory.resolve()), (9,6))
        self.assertAlmostEqual(cam.matrix[0,0], 718.14, 1)

    def test_save_calibration(self):
        directory = pathlib.Path('.')
        cam = Camera()
        cam.calibrate(str(directory.resolve()), (9,6))
        cam.dump_calibration('./test.p')
        self.assertTrue(pathlib.Path('./test.p').exists())
        os.remove(pathlib.Path('./test.p'))

    def test_load_calibration(self):
        directory = pathlib.Path('.')
        cam1 = Camera()
        cam1.calibrate(str(directory.resolve()), (9,6))
        cam1.dump_calibration('./test.p')

        cam2 = Camera()
        cam2.load_calibration('./test.p')
        self.assertAlmostEqual(cam2.matrix[0,0], 718.14, 1)
        os.remove(pathlib.Path('./test.p'))


if __name__ == '__main__':
    unittest.main()
