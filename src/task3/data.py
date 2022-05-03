from pathlib import Path
import tensorflow as tf

# your path to the IAM folder
local_path_to_iam = "C:\\Users\\muell\\Desktop\\HWR\\Task 3\Data\\"
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
    :return: tf.data.Dataset (image, (filename, label))
    """

    # list of filenames & labels as strings
    raw_files = list(data_dict.keys())
    raw_labels = list(data_dict.values())

    # filenames & labels as datasets
    files_ds = tf.data.Dataset.from_tensor_slices(raw_files)
    labels_ds = tf.data.Dataset.from_tensor_slices(raw_labels)
    files_and_labels_ds = tf.data.Dataset.zip((files_ds, labels_ds))

    # absolute img paths as dataset
    dataset = tf.data.Dataset.list_files([str(img_dir.absolute() / f) for f in raw_files])

    # load images from paths
    dataset = dataset.map(lambda x: tf.io.read_file(x))
    dataset = dataset.map(lambda x: tf.io.decode_png(x))
    dataset = dataset.map(lambda x: tf.squeeze(x))

    # combine images, filenames and labels
    dataset = tf.data.Dataset.zip((dataset, files_and_labels_ds))

    return dataset


def test():
    data_dict = load_data_dict()
    dataset = load_dataset(data_dict)

    it = (dataset.as_numpy_iterator())
    for i, e in enumerate(it):
        print(e)
        if i > 3:
            break


if __name__ == "__main__":
    test()
