"""
Train and test a model on the IAM dataset.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import tensorflow as tf

from pathlib import Path

from task3.data import load_data_dict, load_dataset, train_test_split_iam, to_dict, from_dict
from task3.data import tokens_from_text, get_full_token_set
from task3.preprocessing import invert_color, distortion_free_resize, scale_img
from task3.preprocessing import LabelEncoder, LabelPadding
from task3.model import build_LSTM_model, CTCDecodingLayer


# Set path to the IAM folder
#local_path_to_iam = "C:\\Users\\Luca\\Desktop\\HWR"
local_path_to_iam = "C:\\Users\\muell\\Desktop\\HWR\\Task 3\\Data"


def main():
    global local_path_to_iam

    if len(sys.argv) > 1:
        local_path_to_iam = str(sys.argv[1])

    data_dir = Path(local_path_to_iam) / "IAM-data"
    img_dir = data_dir / "img"
    print(f"IAM Path: {data_dir}")

    # Settings
    EPOCHS = 10
    BATCH_SIZE = 24
    LEARNING_RATE = 0.001
    OPTIMIZER = tf.keras.optimizers.RMSprop(LEARNING_RATE)
    METRICS = []

    # Set input dimensions
    image_width = 800
    image_height = 64
    image_channels = 1
    input_shape = (image_width, image_height, image_channels)

    # Load data
    data_dict = load_data_dict(data_dir)
    dataset = load_dataset(data_dict, img_dir)

    # Get tokens
    full_text = "".join(data_dict.values())
    tokens = tokens_from_text(full_text)

    # longest label
    max_label_len = max(list(map(len, data_dict.values())))

    # Prepare label encoding & padding
    label_encoder = LabelEncoder(tokens)
    label_padding = LabelPadding(pad_value=len(tokens), max_len=max_label_len, label_encoder=label_encoder)

    # Pre-process images
    dataset = dataset.map(lambda x, y: (invert_color(x), y))
    dataset = dataset.map(lambda x, y: (distortion_free_resize(x, img_size=(image_width, image_height),
                                                               pad_value=0), y))
    dataset = dataset.map(lambda x, y: (scale_img(x), y))

    # Pre-process labels
    dataset = dataset.map(lambda x, y: (x, label_encoder.encode(y)))
    dataset = dataset.map(lambda x, y: (x, tf.py_function(func=label_padding.add, inp=[y], Tout=tf.int64)))

    # Change to dict format, prepare for input
    dataset = dataset.map(to_dict)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Split data
    train_ds, test_ds = train_test_split_iam(dataset, train_size=0.8, shuffle=True)

    train_model = build_LSTM_model(len(tokens) + 2, image_width)
    train_model.compile(OPTIMIZER)

    final_model = tf.keras.models.Model(
        train_model.get_layer(name="Image").input, train_model.get_layer(name="Output_Softmax").output
    )

    print(train_model.summary())

    history = train_model.fit(train_ds.skip(100), validation_data=train_ds.take(100), epochs=EPOCHS)

    y_pred = train_model.predict(test_ds.take(1))
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.int64)
    for output in y_pred:
        out = tf.gather(output, tf.where(tf.math.not_equal(output, -1)))
        out = tf.strings.reduce_join(label_encoder.decode(out)).numpy().decode('UTF-8')
        print(out)


if __name__ == "__main__":
    main()
