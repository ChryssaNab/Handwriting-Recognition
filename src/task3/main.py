import tensorflow as tf
import os
from task3.data import *
from task3.model import *
from task3.utils import *


# Settings
FORCE_CPU = False
EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = tf.keras.optimizers.RMSprop(LEARNING_RATE)
METRICS = []

if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

#build and train model
model = build_LSTM_model(len(chars) + 1)
train_model(model, train_ds.take(32).batch(32), chars, EPOCHS, OPTIMIZER, METRICS, batch_size=BATCH_SIZE)
pred = test_model(model, test_ds.take(32).batch(32).map(lambda x, y: x), tf.constant(100, shape=(32)), chars)
for p in pred:
    print(p)
