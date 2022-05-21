import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from typing import List, Tuple, Optional, Union


@tf.function
def invert_color(image: tf.Tensor, y: Optional[tf.Tensor] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    image = image * tf.constant(-1, dtype=image.dtype)
    image = tf.add(image, tf.constant(255, dtype=image.dtype))
    if y is None:
        return image
    return image, y


@tf.function
def distortion_free_resize(image: tf.Tensor,
                           img_size: Tuple[int, int] = None,
                           pad_value: int = 255
                           ) -> tf.Tensor:
    """
    Pad and resize image to new size while conserving aspect ratio.

    :param image: image as tensor
    :param img_size: new height and width
    :param pad_value: color of padded areas
    :return: resized image
    """

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
        constant_values=pad_value,
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


@tf.function
def scale_img(image: tf.Tensor, y: Optional[tf.Tensor] = None) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """
    Scale image values from [0, 255] to [0.0, 1.0].

    :param image: image as tensor
    :param y: optional label (ignored by function)
    :return: scaled image, y if passed as arg
    """
    if y is None:
        return tf.cast(image, tf.float32) / 255.
    return tf.cast(image, tf.float32) / 255., y


def remove_filenames(dataset: tf.data.Dataset) -> tf.data.Dataset:
    x = dataset.map(lambda x, f, y: x)
    y = dataset.map(lambda x, f, y: y)
    return tf.data.Dataset.zip((x, y))


class LabelEncoder:
    def __init__(self, tokens: List[str]):
        self.token_list = tokens
        self._lookup = StringLookup(vocabulary=tokens, mask_token=None)

    def encode(self, label: str, unicode_split: bool = True):
        if not unicode_split:
            return self._lookup(label)
        return self._lookup(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    def get_vocabulary(self) -> List[str]:
        return self._lookup.get_vocabulary()


class LabelDecoder:
    pass
