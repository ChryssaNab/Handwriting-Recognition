"""
Data augmentation tools for IAM
"""

import tensorflow as tf
from tensorflow.keras import layers

def data_augment():
    data_augmentation = tf.keras.Sequential([
        # layers.RandomBrightness(-.5,0.5),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.5)
    ])
    return data_augmentation

# TODO: add data augmentation tools
