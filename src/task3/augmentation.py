"""
Data augmentation tools for IAM
"""

from tensorflow.keras import Sequential, layers


def data_augment():
    """
    Data augmentation for images.

    :return: augmentation layers as Sequential model
    """

    data_augmentation = Sequential([
        layers.RandomTranslation(0.1, 0.1, fill_mode="constant", fill_value=0.0),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.5),
        layers.GaussianNoise(0.1),
    ], name="Data_Augmentation")
    return data_augmentation
