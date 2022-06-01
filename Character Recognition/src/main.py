""" This module implements the end-to-end pipeline. """

import glob
import time
import os

from matplotlib import image
from tensorflow import keras

from n_grams import getProbs
from preprocessing import preprocessing
from segmentation import segment
from predict import predict
from opts import parse_opts
from utils import makeDir, removeDir, save_image


def main():
    opt = parse_opts()
    opt.segmentation_path = os.path.join(opt.result_path, opt.segmentation_path)
    opt.transcript_path = os.path.join(opt.result_path, opt.transcript_path)

    # Remove old instance of results if exist
    removeDir(opt.result_path)

    # Collect images
    data_dir = os.path.join(opt.data_path, "*" + opt.extension)
    images = glob.glob(data_dir)
    # Keep only binary images
    binary_images = [name for name in images if name.endswith("binarized.jpg")]

    start_time = time.time()

    # Create bigram language model of Hebrew characters'
    bigram_model = getProbs(opt.ngrams_freq)

    # Load trained model
    model = keras.models.load_model('./training/model.h5')

    for file in binary_images:
        # Load single image
        img = image.imread(file)
        
        # Preprocess img according to intensity values
        img = preprocessing(img)
        
        # Segment preprocessed img
        segment_output_file = os.path.join(opt.segmentation_path, os.path.basename(file).split(opt.extension)[0])
        makeDir(segment_output_file)
        segment(segment_output_file, img)
        
        # Recognize segmented characters
        makeDir(opt.transcript_path)
        predict(segment_output_file, opt.transcript_path, model, bigram_model)

    # Measure elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: " + str(elapsed_time))


if __name__ == '__main__':
    main()
