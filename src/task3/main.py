"""
Train and test a model on the IAM dataset.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

from task3.data import load_data_dict, load_dataset, train_test_split_iam
from task3.data import tokens_from_text, full_token_set
from task3.preprocessing import invert_color, distortion_free_resize, scale_img
from task3.model import build_LSTM_model
from task3.utils import train_model, test_model


# Set path to the IAM folder
#local_path_to_iam = "C:\\Users\\Luca\\Desktop\\HWR"
local_path_to_iam = "C:\\Users\\muell\\Desktop\\HWR\\Task 3\\Data"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        local_path_to_iam = str(sys.argv[1])
        print(sys.argv[1])

    data_dir = Path(local_path_to_iam) / "IAM-data"
    img_dir = data_dir / "img"
    print(f"IAM Path: {data_dir}")

    # Settings
    EPOCHS = 2
    BATCH_SIZE = 24
    LEARNING_RATE = 0.001
    OPTIMIZER = tf.keras.optimizers.RMSprop(LEARNING_RATE, clipvalue=1.0)
    METRICS = []
    LOGIT_LEN = 200

    # Set input dimensions
    image_width = 800
    image_height = 64
    image_channels = 1
    input_shape = (image_width, image_height, image_channels)

    # Load data
    data_dict = load_data_dict(data_dir)
    dataset = load_dataset(data_dict, img_dir)

    # Pre-process data
    dataset = dataset.map(invert_color)
    dataset = dataset.map(lambda x, y: (distortion_free_resize(x, img_size=(image_width, image_height), pad_value=0), y))
    dataset = dataset.map(scale_img)

    # Split data
    train_ds, test_ds = train_test_split_iam(dataset, train_size=0.8, shuffle=True)

    # Create list of unique characters for encoding
    full_text = "".join(data_dict.values())
    tokens = tokens_from_text(full_text)

    # build and train model
    model = build_LSTM_model(len(tokens) + 1)

    train_model(model,
                train_ds,
                tokens,
                EPOCHS,
                OPTIMIZER,
                METRICS,
                batch_size=BATCH_SIZE,
                )

    pred = test_model(model,
                      test_ds.map(lambda x, y: x).batch(batch_size=BATCH_SIZE, drop_remainder=True).take(5),
                      tf.constant(LOGIT_LEN, shape=BATCH_SIZE),
                      tokens,
                      )

    for p in pred:
        print(p)
