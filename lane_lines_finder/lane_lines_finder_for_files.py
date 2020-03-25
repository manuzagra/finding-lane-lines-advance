from lane_lines_finder.utils import get_image, save_image
from lane_lines_finder.pipeline import Pipeline
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
            processed_img = self.pipeline(img)
            save_image(processed_img, out_image)

    def process_images_directory(self, in_dir, out_dir):
        pass

    def process_video(self, in_video, out_video):
        pass

    def process_videos_directory(self, in_dir, out_dir):
        pass


if __name__ == '__main__':

    pipeline = Pipeline()

    camera = cam.Camera()

    pipeline.append(step.Grayscale())
    pipeline.append(step.SaveImage(directory='../test_images_output', name='gray.jpg'))
    pipeline.append(step.Threshold(min=0, max=100))
    pipeline.append(step.Binary2Color(color=(255,0,0)))
    pipeline.append(step.SaveImage(directory='../test_images_output', name='binary.jpg'))

    img = get_image('test1.jpg', '../test_images')

    process_img = pipeline.process(img)

    p = LaneLinesFinderForFiles()

    p.calibrate_camera('../camera_cal', (9, 6))
    #p.camera.dump_calibration('../camera_cal/calibration.p')
    p.set_pipeline(llf_pipeline)
    p.process_image('../test_images/test1.jpg', '../test_images_output/test1.jpg')


