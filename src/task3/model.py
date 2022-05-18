"""
Model architectures for IAM
"""

import tensorflow as tf


def build_LSTM_model(n_classes: int, batch_size=32) -> tf.keras.Sequential:
    """
    Model architecture adapted from:
    Handwritten Text Recognition in Historical Documents, p.38
    https://web.archive.org/web/20210814184909id_/https://repositum.tuwien.at/bitstream/20.500.12708/5409/2/Scheidl%20Harald%20-%202018%20-%20Handwritten%20text%20recognition%20in%20historical%20documents.pdf

    :param n_classes: number of classes to predict (i.e. number of characters)
    :param batch_size: training batch size
    :return: the model as keras Sequential model
    """


    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(800, 64, 1), batch_size=batch_size, name="Input"),
        tf.keras.layers.Conv2D(64, 5, padding="same", activation="relu", name="Conv_1"),
        tf.keras.layers.MaxPool2D(padding="same", name="MaxPool_1"),
        tf.keras.layers.Conv2D(128, 5, padding="same", activation="relu", name="Conv_2"),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same", name="MaxPool_2"),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="Conv_3"),
        tf.keras.layers.MaxPool2D(padding="same", name="MaxPool_3"),
        tf.keras.layers.BatchNormalization(name="BatchNorm_1"),
        tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="Conv_4"),
        tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="Conv_5"),
        tf.keras.layers.MaxPool2D(padding="same", name="MaxPool_4"),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="Conv_6"),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same", name="MaxPool_5"),
        tf.keras.layers.BatchNormalization(name="BatchNorm_2"),
        tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="Conv_7"),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same", name="MaxPool_6"),
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x), name="Collapse"),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512), name="BiDir_LSTM"),
        tf.keras.layers.Dense(n_classes * 100, activation=None, name="Output_Dense"),
        tf.keras.layers.Reshape((100, n_classes), name="Output_Reshape"),
        tf.keras.layers.Softmax(axis=-1, name="Output_Softmax"),
    ])

    return model
