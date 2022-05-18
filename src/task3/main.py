"""
Train and test a model on the IAM dataset.
"""

import tensorflow as tf
import os
FORCE_CPU = True
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from task3.data import *
from task3.model import *
from task3.utils import *


# Settings
EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.001
OPTIMIZER = tf.keras.optimizers.RMSprop(LEARNING_RATE)
METRICS = []

# Load data
data_dict = load_data_dict()
dataset = load_dataset(data_dict)

image_width = 800
image_height = 64

# Preprocess data
dataset = dataset.apply(remove_filenames)
dataset = dataset.map(lambda x, y: (scale_img(x), y))
dataset = dataset.map(lambda x, y: (distortion_free_resize(x, img_size=(image_width, image_height)), y))

# Split data
train_ds, test_ds = train_test_split_iam(dataset, train_size=0.8, shuffle=True)

# Create list of unique characters for encoding
full_text = "".join(data_dict.values())
chars = sorted(list(set(full_text)))

# build and train model
model = build_LSTM_model(len(chars) + 1)

train_model(model,
            train_ds.take(256),
            chars,
            EPOCHS,
            OPTIMIZER,
            METRICS,
            batch_size=BATCH_SIZE,
            )

pred = test_model(model,
                  test_ds.map(lambda x, y: x).batch(batch_size=BATCH_SIZE, drop_remainder=True).take(1),
                  tf.constant(100, shape=BATCH_SIZE),
                  chars,
                  )

for p in pred:
    print(p)
