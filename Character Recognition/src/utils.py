""" This module implements functions used in multiple scripts. """

import os
import shutil
from PIL import Image


def makeDir(p) -> None:

    """ Creates new directory recursively if not exists.
    :param p: Candidate directory path
    """
    
    if p and (not os.path.exists(p)):
        os.makedirs(p)


def removeDir(p) -> None:

    """ Removes old instance of results if exists.
    :param p: Candidate directory path
    """
    
    if p and (os.path.exists(p)):
        shutil.rmtree(p)


def save_image(image, path) -> None:
    
    """ Saves image in device.
    :param image: A np.ndarray with image
    :param path: Path of output directory
    """

    im = Image.fromarray(image)
    im.save(path + ".jpg")
