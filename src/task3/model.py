import tensorflow as tf


def build_LSTM_model() -> tf.keras.Sequential:
    # HWR in Historical Documents, p.38
    # https://web.archive.org/web/20210814184909id_/https://repositum.tuwien.at/bitstream/20.500.12708/5409/2/Scheidl%20Harald%20-%202018%20-%20Handwritten%20text%20recognition%20in%20historical%20documents.pdf
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(800, 64, 1)),
        tf.keras.layers.Conv2D(64, 5, padding="same"),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.Conv2D(128, 5, padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same"),
        tf.keras.layers.Conv2D(128, 3, padding="same"),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 3, padding="same"),
        tf.keras.layers.Conv2D(256, 3, padding="same"),
        tf.keras.layers.MaxPool2D(padding="same"),
        tf.keras.layers.Conv2D(512, 3, padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, 3, padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same"),
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return model
