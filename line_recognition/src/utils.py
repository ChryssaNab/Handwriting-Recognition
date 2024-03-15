"""
Utilities for IAM
"""

import time
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from typing import NamedTuple, Dict, Union
from collections import namedtuple


def get_parser() -> argparse.ArgumentParser:
    """
    Parser for IAM main.
    Takes IAM-path, train/test mode and debug mode as args.

    :return: parser with args
    """
    parser = argparse.ArgumentParser(description="args for IAM main")
    parser.add_argument('path', type=str, nargs='?', default=None, help="path to 'img' (test) or 'IAM-data' (train)")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test',
                        help="train or test model (default test)")
    parser.add_argument('--debug', action='store_true', default=False, help="use debug mode (default False)")
    parser.add_argument('--final', action='store_true', default=False, help="train model on the whole dataset "
                                                                            "(default False)")
    return parser


def make_dirs(root_dir: Path) -> NamedTuple:
    """
    Create folders for results at root_dir.

    :param root_dir: folder iam_results
    :return: paths to logs, models, checkpoints, & settings
    """
    PathTuple = namedtuple("paths",  "logs model checkpoint settings")
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
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
              f"{duration // 60**2}h {(duration // 60) % 60}m {(duration % 60):.2f}s")

    return _inner


def write_predictions(predictions: Dict[str, str], file_path: Union[str, Path] = None) -> bool:
    """
    Write input file names and predictions to txt file.

    :param predictions: dict containing {file name: predictions}
    :param file_path: path to output file
    :return: False if IOError
    """
    try:
        if file_path is None:
            file_path = "predictions.txt"
        with open(str(file_path), "w") as f:
            for img_file, pred in predictions.items():
                f.write(f"{img_file}\n{pred}\n\n")
        return True

    except IOError:
        return False


def show_sample(X: tf.Tensor, y: str) -> None:
    """
    Show image and label with matplotlib.

    :param X: image as tensor
    :param y: label as string
    """
    plt.imshow(tf.transpose(tf.image.flip_left_right(X), [1, 0, 2]), cmap='Greys')
    plt.title(y)
    plt.show()
