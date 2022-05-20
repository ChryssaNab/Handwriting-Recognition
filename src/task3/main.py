"""
Train and test a model on the IAM dataset.
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
import tensorflow as tf
from task3.data import *
from task3.model import *
from task3.utils import *


# Settings
EPOCHS = 2
BATCH_SIZE = 24
LEARNING_RATE = 0.001
OPTIMIZER = tf.keras.optimizers.RMSprop(LEARNING_RATE, clipvalue=1.0)
METRICS = []

# Load data
data_dict = load_data_dict()
dataset = load_dataset(data_dict)

image_width = 800
image_height = 64
image_channels = 1
input_shape = (image_width, image_height, image_channels)

# Preprocess data
dataset = dataset.apply(remove_filenames)
dataset = dataset.map(lambda x, y: (invert_color(x), y))
dataset = dataset.map(lambda x, y: (distortion_free_resize(x, img_size=(image_width, image_height), pad_value=0), y))
dataset = dataset.map(lambda x, y: (scale_img(x), y))

# Split data
train_ds, test_ds = train_test_split_iam(dataset, train_size=0.8, shuffle=True)

# Create list of unique characters for encoding
full_text = "".join(data_dict.values())
chars = sorted(list(set(full_text)))

# build and train model
model = build_LSTM_model(len(chars) + 1)

train_model(model,
            train_ds,
            chars,
            EPOCHS,
            OPTIMIZER,
            METRICS,
            batch_size=BATCH_SIZE,
            )

pred = test_model(model,
                  test_ds.map(lambda x, y: x).batch(batch_size=BATCH_SIZE, drop_remainder=True).take(5),
                  tf.constant(200, shape=BATCH_SIZE),
                  chars,
                  )

for p in pred:
    print(p)
