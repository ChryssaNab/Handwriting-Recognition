"""
Data loading and processing utilities for IAM
"""

import tensorflow as tf
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split


# Set path to the IAM folder
local_path_to_iam = "C:\\Users\\muell\\Desktop\\HWR\\Task 3\\Data"
data_dir = Path(local_path_to_iam) / "IAM-data"
img_dir = data_dir / "img"


def load_data_dict(data_dir: Path = data_dir) -> dict:
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


def load_dataset(data_dict: dict) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from image names and labels.
    Data is not preprocessed.

    :param data_dict: dict {filename: label}
    :return: tf.data.Dataset (image, filename, label)
    """

    # List of filenames & labels as strings
    raw_files = list(data_dict.keys())
    raw_labels = list(data_dict.values())

    # Filenames & labels as datasets
    files = tf.data.Dataset.from_tensor_slices(raw_files)
    labels = tf.data.Dataset.from_tensor_slices(raw_labels)

    # Absolute img paths as dataset
    images = tf.data.Dataset.from_tensor_slices([str(img_dir.absolute() / f) for f in raw_files])

    # Load images from paths
    images = images.map(lambda x: tf.io.read_file(x))
    images = images.map(lambda x: tf.io.decode_png(x))
    #images = images.map(lambda x: tf.squeeze(x))

    # Combine images, filenames and labels
    dataset = tf.data.Dataset.zip((images, files, labels))

    return dataset


def distortion_free_resize(image: tf.Tensor, img_size: Tuple[int, int]) -> tf.Tensor:
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check the amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


def scale_img(image: tf.Tensor) -> tf.Tensor:
    return tf.cast(image, tf.float32) / 255.


def preprocess(image: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.mobilenet.preprocess_input(tf.cast(image, tf.float32))


def remove_filenames(dataset: tf.data.Dataset) -> tf.data.Dataset:
    x = dataset.map(lambda x, f, y: x)
    y = dataset.map(lambda x, f, y: y)
    return tf.data.Dataset.zip((x, y))


def train_test_split_iam(dataset: tf.data.Dataset,
                         train_size=0.8,
                         shuffle=False,
                         ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    assert 0.0 < train_size < 1.0, f"Parameter train_size not a float in range (0, 1): {train_size}"
    ds_size = int(dataset.cardinality())
    test_size = 1.0 - train_size
    n_test_samples = int(ds_size * test_size)
    if shuffle:
        dataset.shuffle()
    test_dataset = dataset.take(n_test_samples)
    train_dataset = dataset.skip(n_test_samples)
    return train_dataset, test_dataset


def split_data(i, l):
    x_tr, x_te, y_tr, y_te = train_test_split(i, l, train_size=0.95)
    return x_tr, x_te, y_tr, y_te


def test(image_width: int, image_height: int):
    data_dict = load_data_dict()
    dataset = load_dataset(data_dict)

    # preprocessing example
    # preprocessing using inbuilt function
    dataset = dataset.map(lambda x, f, y: (preprocess(x), f, y))
    # padding
    dataset = dataset.map(lambda x, f, y: (distortion_free_resize(x, img_size=(image_width, image_height)), f, y))
    dataset = dataset.apply(remove_filenames)

    it = (dataset.as_numpy_iterator())
    for i, e in enumerate(it):
        #print(e)
        pass
        if i > 3:
            break

    return dataset


if __name__ == "__main__":

    image_width = 512
    image_height = 64
    #data = test(image_width, image_height)
    data = test(image_width, image_height)
    images, labels = zip(*data)
    x_train, x_test, y_train, y_test = split_data(images, labels)

    print(len(x_train))
    print(len(y_train))
    print(len(x_test))
    print(len(y_test))

    import matplotlib.pyplot as plt
    it = data.as_numpy_iterator()
    for d in it:
        plt.imshow(d[0])
        plt.title(d[1])
        plt.show()
        #print(d[0].shape)
