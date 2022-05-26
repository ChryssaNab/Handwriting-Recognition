"""
Data loading utilities for IAM
"""

import tensorflow as tf
from typing import Tuple, List, Dict, Union
from pathlib import Path


def load_data_dict(data_dir: Path) -> dict:
    """
    Returns dictionary with filenames as keys and labels as values.

    :param data_dir: path to 'IAM-data'
    :return: dict {filename: label}
    """
    with open(str(data_dir / "iam_lines_gt.txt"), "r") as f:
        labels_file = f.read()

    labels_file = labels_file.split("\n")
    data_dict = dict()
    img = None

    for line in labels_file:
        if line.endswith(".png"):
            img = line
        elif len(line):
            data_dict[img] = line
            img = None

    return data_dict


def load_dataset(data_dict: Dict[str, str], img_dir: Path, return_filenames: bool = False) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from image names and labels.
    Data is not preprocessed.

    :param data_dict: dict {filename: label}
    :param img_dir: path to IAM images
    :param return_filenames: if 'True' return tf.data.Dataset (image, filename, label)
    :return: tf.data.Dataset (image, label)
    """

    # List of filenames & labels as strings
    raw_files = list(data_dict.keys())
    raw_labels = list(data_dict.values())

    # Labels as dataset
    labels = tf.data.Dataset.from_tensor_slices(raw_labels)

    # Absolute img paths as dataset
    images = tf.data.Dataset.from_tensor_slices([str(img_dir.absolute() / f) for f in raw_files])

    # Load images from paths
    images = images.map(lambda x: tf.io.read_file(x))
    images = images.map(lambda x: tf.io.decode_png(x))

    if not return_filenames:
        dataset = tf.data.Dataset.zip((images, labels))
        return dataset

    # Combine images, filenames and labels
    files = tf.data.Dataset.from_tensor_slices(raw_files)
    dataset = tf.data.Dataset.zip((images, files, labels))

    return dataset


def split_data_dict(data_dict: Dict[str, str],
                    test_split: float = 0.85,
                    validation_split: float = 0.9,
                    shuffle: bool = False,
                    ) -> Tuple[dict, dict, dict]:
    """
    Split data dict into train, validation and test set.

    :param data_dict: full IAM data dict
    :param test_split: relative test size (taken from whole dataset)
    :param validation_split: relative validation set size (taken from train dataset)
    :param shuffle: shuffle dict after test / before validation split
    :return: tuple (train_data, validation_data, test_data)
    """

    import random
    assert 0.0 < test_split < 1.0, f"Expected train_size to be a float in range (0.0, 1.0), got {test_split} instead."

    items = list(data_dict.items())
    test_idx = int(test_split * len(items))
    train_data, test_data = items[:test_idx], items[test_idx:]
    if shuffle:
        random.shuffle(train_data)
    validation_idx = int(validation_split * len(train_data))
    train_data, validation_data = train_data[:validation_idx], train_data[validation_idx:]
    train_data, validation_data, test_data = dict(train_data), dict(validation_data), dict(test_data)

    return train_data, validation_data, test_data


def train_test_split(dataset: tf.data.Dataset,
                     train_size: float = 0.8,
                     shuffle: bool = False,
                     ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split IAM dataset into separate train and test sets.

    :param dataset: full IAM dataset
    :param train_size: float (0.0, 1.0) determining test split
    :param shuffle: if 'True' shuffle dataset before split
    :return: train and test dataset
    """

    assert 0.0 < train_size < 1.0, f"Expected train_size to be a float in range (0.0, 1.0), got {train_size} instead."
    ds_size = dataset.cardinality()
    test_size = tf.constant(1.0 - train_size, dtype=tf.float32)
    n_test_samples = tf.cast(test_size * tf.cast(ds_size, dtype=tf.float32), dtype=ds_size.dtype)
    if shuffle:
        dataset.shuffle(ds_size)
    test_dataset = dataset.take(n_test_samples)
    train_dataset = dataset.skip(n_test_samples)
    return train_dataset, test_dataset


def filter_labels(sample: Union[tf.Tensor, Dict[str, tf.Tensor]], *args) -> tf.Tensor:
    """
    Remove labels from dataset for predictions.

    :param sample: X as tensor or dict with X as 'Image' and y as 'Label'
    :param args: y as tensor if X is a tensor
    :return: X as tensor
    """
    if isinstance(sample, dict):
        return sample["Image"]
    return sample


def to_dict(x: tf.Tensor, y: tf.Tensor, x_key: str = "Image", y_key: str = "Label") -> Dict[str, tf.Tensor]:
    """
    Combine two tensors into a dict.

    :param x: x tensor
    :param y: y tensor
    :param x_key: name of x category
    :param y_key: name of y category
    :return: dict {x_key: x, y_key: y}
    """
    return {x_key: x, y_key: y}


def from_dict(d: dict, x_key: str = "Image", y_key: str = "Label") -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Undo to_dict by extracting two tensors from dict.

    :param d: dict with (at least) two tensors
    :param x_key: name of x category
    :param y_key: name of y category
    :return: tuple (x, y)
    """
    return d[x_key], d[y_key]


def tokens_from_text(text: str) -> List[str]:
    """
    Extract unique characters from a string and return them as a sorted list.

    :param text: string containing tokens
    :return: sorted list of unique tokens
    """
    return sorted(list(set(text)))


def get_full_token_set() -> List[str]:
    """
    Get a token set with all common characters.

    :return: list of tokens
    """
    tokens = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>',
              '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', 'a',  'b', 'c', 'd', 'e', 'f', 'g', 'h',
              'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',  'v', 'w', 'x', 'y', 'z', 'A', 'B',
              'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
              'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']

    return tokens
