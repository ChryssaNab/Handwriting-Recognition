import glob
import time
import os
from matplotlib import image
from opts import parse_opts
from utils import makeDir, removeDir
from segmentation import *


def main():
    opt = parse_opts()

    opt.preprocessing_path = os.path.join(opt.result_path, opt.preprocessing_path)
    opt.segmentation_path = os.path.join(opt.result_path, opt.segmentation_path)

    # Remove old instance of results if exist
    removeDir(opt.result_path)

    # Load images
    data_dir = opt.data_path + "*" + opt.extension
    images = glob.glob(data_dir)
    # Keep only RGB images
    rgb_images = [name for name in images if name.endswith("R01-binarized.jpg")]  # to be changed to "R01.jpg"

    start_time = time.time()

    for file in rgb_images:
        img = image.imread(file)

        # Preprocess img | :return: np.ndarray binarized images

        segment_output_file = os.path.join(opt.segmentation_path, os.path.basename(file).split(opt.extension)[0])
        makeDir(segment_output_file)
        # Segment preprocessed img
        words = segment(segment_output_file, img)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: " + str(elapsed_time))


if __name__ == '__main__':
    main()
