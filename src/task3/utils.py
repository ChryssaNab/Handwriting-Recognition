"""
Utilities for IAM
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import NamedTuple
from collections import namedtuple


def show_sample(X: tf.Tensor, y: tf.Tensor) -> None:
    plt.imshow(tf.transpose(tf.image.flip_left_right(X), [1, 0, 2]), cmap='Greys')
    plt.title(y)
    plt.show()


def make_dirs(root_dir: Path) -> NamedTuple:
    """
    Create folders for results at root_dir.

    :param root_dir: folder iam_results
    :return: paths to logs, models, checkpoints
    """
    import time
    PathTuple = namedtuple("paths",  "logs model checkpoint")
    timestamp = time.strftime("_%Y_%m_%d-%H_%M_%S")
    run_id = "run" + timestamp
    logs = root_dir / "logs" / run_id
    logs.mkdir(exist_ok=True, parents=True)
    model = root_dir / "models" / run_id
    model.mkdir(exist_ok=True, parents=True)
    checkpoint = root_dir / "checkpoints" /run_id
    checkpoint.mkdir(exist_ok=True, parents=True)
    return PathTuple(logs, model, checkpoint)

# TODO: parameter search, crossvalidation
