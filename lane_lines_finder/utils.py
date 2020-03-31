"""
This file contains functions that may be useful in the whole project.
"""

import cv2
import pathlib


def get_image(fname, directory=''):
    """
    Read an image from a directory
    :param fname: name of the image or compelte path
    :param directory: directory to look for the image, not needed if fname has the complete path
    :return: np.array containing the image
    """
    input = pathlib.Path(directory).joinpath(fname).resolve()
    # if it does not exist return None
    if not input.exists():
        return None
    # Read the image, it can be an image or None if the file is not an image
    img = cv2.imread(str(input))
    return img


def save_image(image, fname, directory=''):
    """
    Save an image into disk
    :param image: np.array
    :param fname: name of the image or compelte path
    :param directory: directory to look for the image, not needed if fname has the complete path
    """
    output = pathlib.Path(directory).joinpath(fname).resolve()
    cv2.imwrite(str(output), image)


def images_generator(directory):
    """
    Reads all the images in a directory
    :param directory: directory to look for the image
    :return:
    """
    for pfile in pathlib.Path(directory).iterdir():
        # Read the image
        img = get_image(pfile)
        if img is None:  # the file is not an image
            continue
        yield img
