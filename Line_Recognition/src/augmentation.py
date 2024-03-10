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
        layers.RandomRotation(0.02),
        layers.RandomTranslation(0.02, 0.05, fill_mode="constant", fill_value=0.0),
        layers.RandomContrast(0.8),
        layers.GaussianNoise(0.02),
    ], name="Data_Augmentation")
    return data_augmentation
