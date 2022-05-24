"""
Settings for IAM main
"""

import tensorflow as tf


def get_lstm_settings(debug: bool = False) -> dict:
    """
    Settings for the LSTM model.

    :param debug: use debug settings
    :return: dict with settings
    """

    settings = {

        # Training
        "epochs":           10,
        "batch_size":       24,
        "learning_rate":    0.001,
        "optimizer":        tf.keras.optimizers.RMSprop,

        # Image size
        "image_width":      800,
        "image_height":     64,

        # Label length
        "max_label_length": 80,

        # Dataset
        "test_split":       0.85,
        "validation_split": 0.9,

        # CTC
        "ctc_blank":        (-1),

        # IDs
        "model_name":       "LSTM_model",
    }

    if debug:
        settings["epochs"] = 1
        settings["model_name"] += "_debug"

    return settings