import cv2
import pathlib


def get_image(fname, directory=''):
    input = pathlib.Path(directory).joinpath(fname).resolve()
    if not input.exists():
        return None
    # Read the image, it can be an image or None if the file is not an image
    img = cv2.imread(str(input))
    return img


def save_image(image, fname, directory=''):
    output = pathlib.Path(directory).joinpath(fname).resolve()
    cv2.imwrite(str(output), image)


def images_generator(directory):
    for pfile in pathlib.Path(directory).iterdir():
        # Read the image
        img = get_image(pfile)
        if img is None:  # the file is not an image
            continue
        yield img


if __name__ == '__main__':
    pass
