"""
Utilities for IAM
"""

import time
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import NamedTuple
from collections import namedtuple


def make_dirs(root_dir: Path) -> NamedTuple:
    """
    Create folders for results at root_dir.

    :param root_dir: folder iam_results
    :return: paths to logs, models, checkpoints
    """
    PathTuple = namedtuple("paths",  "logs model checkpoint settings")
    timestamp = time.strftime("_%Y_%m_%d-%H_%M_%S")
    run_dir = root_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True)
    logs = run_dir / "logs"
    logs.mkdir()
    model = run_dir / "model"
    model.mkdir()
    checkpoint = run_dir / "checkpoint"
    checkpoint.mkdir()
    settings = run_dir / "settings"
    settings.mkdir()
    return PathTuple(logs, model, checkpoint, settings)


def track_time(func):
    """
    Decorator function to track and print function execution time.

    :param func: function to track
    :return: wrapped function
    """

    def _inner(*args, **kwargs):
        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        duration = (end - begin)
        print(f"Total time taken in function '{func.__name__}': "
              f"{duration // 60**2}h {(duration // 60) % 60**2}m {(duration % 60):.2f}s")

    return _inner


def show_sample(X: tf.Tensor, y: str) -> None:
    """
    Show image and label with matplotlib.

    :param X: image as tensor
    :param y: label as string
    """

    plt.imshow(tf.transpose(tf.image.flip_left_right(X), [1, 0, 2]), cmap='Greys')
    plt.title(y)
    plt.show()

# TODO: parameter search, crossvalidation
