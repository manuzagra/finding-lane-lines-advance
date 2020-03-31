import pathlib
from moviepy.editor import VideoFileClip
from lane_lines_finder.utils import get_image, save_image
import lane_lines_finder.self_driving_car as self_driving_car


class LaneLinesFinderForFiles:
    def __init__(self):
        self.pipeline_factory = None
        self.pipeline = None

    def set_pipeline(self, pl):
        self.pipeline = pl

    def set_pipeline_factory(self, plf):
        self.pipeline_factory = plf
        self.reset_pipeline()

    def reset_pipeline(self):
        self.pipeline = self.pipeline_factory()

    def process_image(self, in_image, out_image):
        img = get_image(in_image)
        if img is not None:
            processed_img, _ = self.pipeline.process(img,
                                                  type='image',
                                                  file=pathlib.Path(in_image).name,
                                                  input_directory=pathlib.Path(in_image).parent,
                                                  output_directory=pathlib.Path(out_image).parent)
            save_image(processed_img, out_image)

    def process_images_directory(self, in_dir, out_dir):
        self.reset_pipeline()
        in_path = pathlib.Path(in_dir).resolve()
        out_path = pathlib.Path(out_dir).resolve()
        for f in in_path.glob('*'):
            self.process_image(str(f), str(out_path.joinpath(f.name)))

    def process_frame(self, img):
        processed_img, _ = self.pipeline.process(img, type='video',
                                                  file='',
                                                  input_directory='',
                                                  output_directory='')
        return processed_img

    def process_video(self, in_video, out_video):
        self.reset_pipeline()
        clip1 = VideoFileClip(in_video).subclip(3, 17)
        white_clip = clip1.fl_image(self.process_frame)  # NOTE: this function expects color images!!
        white_clip.write_videofile(out_video, audio=False)

    def process_videos_directory(self, in_dir, out_dir):
        in_path = pathlib.Path(in_dir).resolve()
        out_path = pathlib.Path(out_dir).resolve()
        for f in in_path.glob('*'):
            self.reset_pipeline()
            #try:
            self.process_video(str(f), str(out_path.joinpath(f.name)))
            #except:
            #    pass


if __name__ == '__main__':

    finder = LaneLinesFinderForFiles()

    finder.set_pipeline_factory(self_driving_car.pipeline)

    finder.process_images_directory('../test_images', '../test_images_output')
    #finder.process_video('../test_videos/project_video.mp4', '../test_videos_output/project_video.mp4')





