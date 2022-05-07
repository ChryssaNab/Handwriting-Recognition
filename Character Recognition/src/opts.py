import argparse


def parse_opts():

    parser = argparse.ArgumentParser()

    # |----------------------------------------------- Paths -----------------------------------------------|
    
    parser.add_argument(
        '--data_path',
        default="/Handwriting Recognition/Assignment/Datasets/DSS/image-data/image-data/",
        type=str,
        help='RGB DDS dataset')

    parser.add_argument(
        '--extension',
        default=".jpg",
        type=str,
        help='Extension of test images')

    parser.add_argument(
        '--result_path',
        default="/Handwriting Recognition/Assignment/Methods/Task1/Results",
        type=str,
        help='Results directory path')

    parser.add_argument(
        '--preprocessing_path',
        default="preprocessing_output",
        type=str,
        help='Save images (.jpg) of pre-processing step')

    parser.add_argument(
        '--segmentation_path',
        default="segmentation_output",
        type=str,
        help='Save images (.jpg) of segmentation step')

    # |----------------------------------------- Hyper-parameters -----------------------------------------|

    args = parser.parse_args()

    return args
