"""
Model architectures for IAM
"""

import tensorflow as tf


# TODO: add CTCDecodingLayer

class CTCLayer(tf.keras.layers.Layer):
    """
    CTC loss as layer based on keras documentation:
    https://keras.io/examples/audio/ctc_asr/
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_length = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_length, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_length, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


def build_LSTM_model(n_classes: int, width: int = 800) -> tf.keras.Model:
    """
    Model architecture adapted from:
    Handwritten Text Recognition in Historical Documents, p.38
    https://web.archive.org/web/20210814184909id_/https://repositum.tuwien.at/bitstream/20.500.12708/5409/2/Scheidl%20Harald%20-%202018%20-%20Handwritten%20text%20recognition%20in%20historical%20documents.pdf

    :param n_classes: number of classes to predict (i.e. number of characters), n_classes < 100
    :param width: width of input image
    :return: the model as keras Sequential model
    """

    height, channels = 64, 1
    logit_length = width // 4

    if width % logit_length:
        raise ValueError("input width not divisible by 4")

    input_img = tf.keras.Input(shape=(width, height, channels), name="Image")
    input_label = tf.keras.layers.Input(name="Label", shape=(None,))

    conv = tf.keras.layers.Conv2D(64, 5, padding="same", activation="relu", name="Conv_1")(input_img)
    conv = tf.keras.layers.MaxPool2D(padding="same", name="MaxPool_1")(conv)
    conv = tf.keras.layers.Conv2D(128, 5, padding="same", activation="relu", name="Conv_2")(conv)
    conv = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same", name="MaxPool_2")(conv)
    conv = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="Conv_3")(conv)
    conv = tf.keras.layers.MaxPool2D(padding="same", name="MaxPool_3")(conv)
    conv = tf.keras.layers.BatchNormalization(name="BatchNorm_1")(conv)
    conv = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="Conv_4")(conv)
    conv = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="Conv_5")(conv)
    conv = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same", name="MaxPool_4")(conv)
    conv = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="Conv_6")(conv)
    conv = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same", name="MaxPool_5")(conv)
    conv = tf.keras.layers.BatchNormalization(name="BatchNorm_2")(conv)
    conv = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="Conv_7")(conv)
    conv = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same", name="MaxPool_6")(conv)
    flat = tf.keras.layers.Reshape((logit_length, 512), name="Collapse")(conv)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, activation='tanh', return_sequences=True),
                                         name="BiDir_LSTM")(flat)
    lstm = tf.keras.layers.Dense(n_classes, activation=None, name="Output_Dense")(lstm)
    softmax = tf.keras.layers.Softmax(axis=-1, name="Output_Softmax")(lstm)
    output = CTCLayer(name="CTC_Loss")(input_label, softmax)

    model = tf.keras.models.Model(inputs=[input_img, input_label], outputs=output, name="LSTM_model")

    return model
