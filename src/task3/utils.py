"""
Utilities for IAM
"""

import matplotlib.pyplot as plt
import tensorflow as tf


def show_sample(X: tf.Tensor, y: tf.Tensor):
    plt.imshow(tf.transpose(tf.image.flip_left_right(X), [1, 0, 2]), cmap='Greys')
    plt.title(y)
    plt.show()

# TODO: model checkpoints, tensorboard, parameter search, crossvalidation
