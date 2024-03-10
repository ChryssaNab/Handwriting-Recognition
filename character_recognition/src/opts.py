""" This module includes the path settings. """

import argparse


def parse_opts():

    parser = argparse.ArgumentParser()

    # |----------------------------------------------- Paths -----------------------------------------------|
    
    parser.add_argument(
        '--data_path',
        type=str,
        help='RGB DSS dataset')

    parser.add_argument(
        '--ngrams_freq',
        default="./ngrams_frequencies_withNames.csv",
        type=str,
        help='File of n-grams frequencies')

    parser.add_argument(
        '--extension',
        default=".jpg",
        type=str,
        help='Extension of test images')

    parser.add_argument(
        '--result_path',
        default="./results",
        type=str,
        help='Results directory path')
        
    parser.add_argument(
        '--segmentation_path',
        default="segmentation_output",
        type=str,
        help='Save images (.jpg) of segmentation step')

    parser.add_argument(
        '--transcript_path',
        default="transcript_output",
        type=str,
        help='Save images (.jpg) of pre-processing step')

    args = parser.parse_args()

    return args
