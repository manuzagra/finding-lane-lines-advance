import pathlib
from lane_lines_finder.utils import get_image, save_image
from lane_lines_finder.pipeline import self_driving_car_pipeline
import lane_lines_finder.steps as step
import lane_lines_finder.camera as cam


class LaneLinesFinderForFiles:
    def __init__(self):
        self.pipeline = None

    def set_pipeline(self, pl):
        self.pipeline = pl

    def process_image(self, in_image, out_image):
        img = get_image(in_image)
        if img is not None:
            processed_img = self.pipeline.process(img,
                                                  file=pathlib.Path(in_image).name,
                                                  input_directory=pathlib.Path(in_image).parent,
                                                  output_directory=pathlib.Path(out_image).parent)
            # TODO uncomment this line
            # save_image(processed_img, out_image)

    def process_images_directory(self, in_dir, out_dir):
        in_path = pathlib.Path(in_dir).resolve()
        out_path = pathlib.Path(out_dir).resolve()
        for f in in_path.glob('*'):
            self.process_image(str(f), str(out_path.joinpath(f.name)))
        pass

    def process_video(self, in_video, out_video):
        pass

    def process_videos_directory(self, in_dir, out_dir):
        pass


if __name__ == '__main__':

    finder = LaneLinesFinderForFiles()

    finder.set_pipeline(self_driving_car_pipeline())

    finder.process_images_directory('../test_images', '../test_images_output')

    i = 0




