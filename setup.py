from setuptools import setup

setup(name='lane_lines_finder',
      version='0.1',
      description='Find lane lines in video and images.',
      url='http://github.com/manuzagra/finding-lane-lines-advance',
      author='manuzagra',
      author_email='manuzagra@gmail.com',
      license='',
      packages=['lane_lines_finder',],
      install_requires=['numpy', 'matplotlib'],  # + 'cv2'
      zip_safe=False)
