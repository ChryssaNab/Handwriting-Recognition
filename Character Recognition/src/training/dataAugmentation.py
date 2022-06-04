""" This module implements the augmentation of training images. """

import random
import skimage.transform
import numpy as np
import cv2


def dilation_erosion(img):

    """ Dilates or erodes input image.
    :param img: Input image
    :return: Transformed image
    """

    i = random.randint(-7, 1)
    kernel = np.ones((2, 2), np.uint8)
    if i == 0:
        return img
    elif i > 0:
        img = cv2.erode(img, kernel, iterations=i)
    elif i < 0:
        img = cv2.dilate(img, kernel, iterations=-i)
    return img


def rotation(img):

    """ Rotates input image.
    :param img: Input image
    :return: Transformed image
    """

    degrees = random.randint(-40, 40)
    # Rotate at degrees angle and pad with whitespace
    img = skimage.transform.rotate(img, angle=degrees, resize=False, cval=1)
    # Transform back to uint8-type
    img = img * 255
    img = img.astype(np.uint8)
    return img


def addBox(img, x_size, y_size):

    """ Adds random box to input image.
    :param img: Input image
    :param x_size: The width of the box
    :param y_size: The height of the box
    :return: Transformed image
    """

    x = random.randint(0, img.shape[0] - x_size)
    y = random.randint(0, img.shape[1] - y_size)
    for i in range(0, x_size):
        for j in range(0, y_size):
            img[x + i][y + j] = 0
    return img


def erasing(img, prob):

    """ Erases input image.
    :param img: Input image
    :param prob: The erasing probability
    :return: Transformed image
    """

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if random.random() < prob:
                img[i][j] = 255
    return img


def crop(img):

    """ Crops input image.
    :param img: Input image
    :return: Transformed image
    """

    crop_left = random.randint(1, 4)
    crop_right = random.randint(1, 4)
    crop_bottom = random.randint(1, 4)
    crop_top = random.randint(1, 4)
    img = img[0 + crop_left:img.shape[0] - crop_right, 0 + crop_bottom:img.shape[1] - crop_top]
    return img


def expand(img):

    """ Expands input image.
    :param img: Input image
    :return: Transformed image
    """

    img = cv2.copyMakeBorder(img, random.randint(0, 5), random.randint(0, 5), random.randint(0, 5),
                             random.randint(0, 5), cv2.BORDER_REPLICATE)
    return img


def augmentation(img):

    """ Augments input image using different transformations.
    :param img: Input image
    :return: Transformed image
    """

    probs = [0.5, 0.2, 0.8, 0.2, 0.1, 0.3]

    if random.random() < probs[1]:
        img = expand(img)
    if random.random() < probs[5]:
        img = addBox(img, 5, 5)
    elif random.random() < probs[2]:
        img = crop(img)
    if random.random() < probs[0]:
        img = rotation(img)
    if random.random() < probs[4]:
        img = dilation_erosion(img)
    if random.random() < probs[3]:
        img = erasing(img, 0.3)

    return img
