"""
Model architectures for IAM
"""

import tensorflow as tf
from typing import Tuple, List


class CTCLossLayer(tf.keras.layers.Layer):
    """
    CTC loss as layer based on keras documentation:
    https://keras.io/examples/audio/ctc_asr/
    """

    def __init__(self, name: str = "CTC_Loss", **kwargs) -> None:
        super().__init__(name=name, trainable=False, **kwargs)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, X: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Compute CTC loss from label and model output.

        :param X: tuple (label, model output)
        :return:
        """

        y_true, y_pred = X
        batch_length = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_length, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_length, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred


class CTCDecodingLayer(tf.keras.layers.Layer):
    """
    CTC decoder as layer:
    https://docs.w3cub.com/tensorflow~python/tf/keras/backend/ctc_decode.html
    """

    def __init__(self, name: str = "CTC_Decoding", **kwargs) -> None:
        super().__init__(name=name, trainable=False, **kwargs)
        self.decode_fn = tf.keras.backend.ctc_decode

    def call(self, y_pred: tf.Tensor) -> List[List[int]]:
        """
        Decode the input with a CTC decoder.

        :param y_pred: model output (batch_size, logits, n_chars)
        :return: tensor with decoded predictions
        """

        batch_length = tf.cast(tf.shape(y_pred)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_length), dtype="int64")

        output = self.decode_fn(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
        return output[0][0]


def build_LSTM_model(n_classes: int, width: int = 800) -> tf.keras.Model:
    """
    Model architecture adapted from:
    Handwritten Text Recognition in Historical Documents, p.38
    https://web.archive.org/web/20210814184909id_/https://repositum.tuwien.at/bitstream/20.500.12708/5409/2/Scheidl%20Harald%20-%202018%20-%20Handwritten%20text%20recognition%20in%20historical%20documents.pdf

    Uses n_classes + 2 outputs:
    https://git.io/J0eXP

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
                                         name="BiDir_LSTM_1")(flat)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, activation='tanh', return_sequences=True),
                                         name="BiDir_LSTM_2")(lstm)
    lstm = tf.keras.layers.Dense(n_classes + 2, activation=None, name="Output_Dense")(lstm)
    softmax = tf.keras.layers.Softmax(axis=-1, name="Output_Softmax")(lstm)
    output = CTCLossLayer(name="CTC_Loss")((input_label, softmax))
    output = CTCDecodingLayer(name="CTC_Decoding")(output)

    model = tf.keras.models.Model(inputs=[input_img, input_label], outputs=output, name="LSTM_model")

    return model


def build_model(n_classes, image_width = 800, image_height = 64):
    # Inputs to the model
    input_img = tf.keras.Input(shape=(image_width, image_height, 1), name="Image")
    labels = tf.keras.layers.Input(name="Label", shape=(None,))

    # First conv block.
    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = tf.keras.layers.MaxPooling2D((2,2), name="pool1", padding = "same")(x)

    # Second conv block.
    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = tf.keras.layers.MaxPooling2D((2,2), name="pool2", padding = "same")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((image_width // 4), (image_height // 4) * 512)
    x = tf.keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True, dropout=0.25))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.25))(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = tf.keras.layers.Dense(n_classes+2, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLossLayer(name="ctc_loss")((labels, x))
    output = CTCDecodingLayer(name="CTC_Decoding")(output)

    # Define the model.
    model = tf.keras.models.Model(inputs=[input_img, labels], outputs=output, name="handwriting_recognizer")
    # Optimizer.
    #opt = tf.keras.optimizers.Adam()
    # Compile the model and return.
    #model.compile(optimizer=opt)
    return model
