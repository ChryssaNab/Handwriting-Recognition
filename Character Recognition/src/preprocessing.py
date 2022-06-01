""" This module applies preprocessing techniques to images."""
import cv2
import numpy as np


def dilation(img):

    """ Applies dilation operation to input image.
    :param img: Input image
    :return: Image after dilation
    """

    # Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # Apply opening to fill the holes
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    # Apply dilation to input image
    dilated = cv2.dilate(img, kernel, iterations=3)

    return dilated


def erosion(img):

    """ Applies erosion operation to input image.
    :param img: Input image
    :return: Image after erosion
    """

    # Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # Apply opening to fill the holes
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    # Apply erosion and then dilation to input image
    eroded = cv2.erode(img, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    return dilated


def preprocessing(img):

    """ Preprocess input image.
    :param img: Input image
    :return: Preprocessed input image
    """

    # Determine preprocessed step based on the average intensity value of image
    if np.mean(img) > 248:
        img = erosion(img)
    else:
        img = dilation(img)

    return img
    
