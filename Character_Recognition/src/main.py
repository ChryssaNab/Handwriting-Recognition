import glob
import time
import os
import tensorflow as tf

import n_grams
import predict

from matplotlib import image
from opts import parse_opts
from utils import makeDir, removeDir, save_image
from segmentation import segment
from preprocessing import preprocessing


def main():
    opt = parse_opts()

    # Define paths for output files
    opt.segmentation_path = os.path.join(opt.result_path, opt.segmentation_path)
    opt.transcript_path = os.path.join(opt.result_path, opt.transcript_path)
    
    # Remove old instance of results if exist
    removeDir(opt.result_path)

    # Load images
    data_dir = os.path.join(opt.data_path, "*" + opt.extension)
    images = glob.glob(data_dir)
    
    # Keep only binary images
    binary_images = [name for name in images if name.endswith("binarized.jpg")]

    start_time = time.time()

    # Create bigram language model of Hebrew characters'
    bigram_model = n_grams.getProbs(opt.ngrams_freq)
    # Load model
    model = tf.keras.models.load_model('./src/training/model.h5')

    for file in binary_images:

        # Load a single image
        img = image.imread(file)
        # Preprocess img according to intensity values
        img = preprocessing(img)

        # Segment preprocessed img
        segment_output_file = os.path.join(opt.segmentation_path, os.path.basename(file).split(opt.extension)[0])
        makeDir(segment_output_file)
        segment(segment_output_file, img)

        # Recognize segmented characters
        makeDir(opt.transcript_path)
        predict.predict(segment_output_file, opt.transcript_path, model, bigram_model)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: " + str(elapsed_time))


if __name__ == '__main__':
    main()
