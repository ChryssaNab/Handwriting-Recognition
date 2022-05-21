"""
Data loading and processing utilities for IAM
"""

import tensorflow as tf
from typing import Tuple, List, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split


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


@tf.function
def load_dataset(data_dict: dict, img_dir: Path, return_filenames: bool = False) -> tf.data.Dataset:
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

    # Labels as datasets
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





@tf.function
def train_test_split_iam(dataset: tf.data.Dataset,
                         train_size: float = 0.8,
                         shuffle: bool = False,
                         ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Split IAM dataset into separate train and test sets.

    :param dataset: full IAM dataset
    :param train_size: float (0.0, 1.0) determining test split
    :param shuffle: if 'True' shuffle dataset before split
    :return:
    """

    assert 0.0 < train_size < 1.0, f"Expected train_size to be a float in range (0, 1), got {train_size} instead."
    ds_size = dataset.cardinality()
    test_size = tf.constant(1.0 - train_size, dtype=tf.float32)
    n_test_samples = tf.cast(test_size * tf.cast(ds_size, dtype=tf.float32), dtype=ds_size.dtype)
    if shuffle:
        dataset.shuffle(dataset.cardinality())
    test_dataset = dataset.take(n_test_samples)
    train_dataset = dataset.skip(n_test_samples)
    return train_dataset, test_dataset


def split_data(i, l):
    x_tr, x_te, y_tr, y_te = train_test_split(i, l, train_size=0.95)
    return x_tr, x_te, y_tr, y_te


@tf.function
def to_dict(x: tf.Tensor, y: tf.Tensor, x_key: str = "image", y_key: str = "label") -> dict:
    return {x_key: x, y_key: y}


@tf.function
def from_dict(d: dict, x_key: str = "image", y_key: str = "label") -> Tuple[tf.Tensor, tf.Tensor]:
    return d[x_key], d[y_key]


def tokens_from_text(text: str) -> List[str]:
    """
    Extract unique characters from a string and return them as a sorted list.

    :param text: string containing tokens
    :return: sorted list of unique tokens
    """
    return sorted(list(set(text)))


def full_token_set() -> List[str]:
    """
    Get a token set with special characters for label encoding.

    :return: list of tokens
    """

    BLANK = "[CTCblank]"
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    tokens = [BLANK, PAD, SOS, EOS, UNK, '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
              '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', 'a',  'b',
              'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',  'v',
              'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']

    return tokens
