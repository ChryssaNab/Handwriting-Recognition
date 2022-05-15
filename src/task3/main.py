import tensorflow as tf
import os
from task3.data import *
from task3.model import *
from task3.utils import *


# Settings
FORCE_CPU = False
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01
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

# Create list of unique characters for encoding
full_text = "".join(data_dict.values())
full_text = [c for c in full_text]
chars = sorted(list(set(full_text)))

#build and train model
model = build_LSTM_model(len(chars) + 1)
train_model(model, dataset.prefetch(10), chars, EPOCHS, OPTIMIZER, METRICS, batch_size=BATCH_SIZE)
