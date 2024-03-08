""" This module implements the training process. """

import os
import glob
import pickle

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import layers, models
from tensorflow.python.ops.numpy_ops import np_config
from dataAugmentation import augmentation

np_config.enable_numpy_behavior()


def preprocessing(img):

    """ Preprocess training images to fit the network's architecture.
    :param img: The train image to be preprocessed
    :return: The train image after preprocessing
    """

    # Resize image to certain dims (avg dims)
    img = cv2.resize(img, (38, 48), interpolation=cv2.INTER_AREA)
    # Convert to single channel
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    # Binarize image for threshold=127
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Normalize image to [0,1]
    im = thresh1 / 255.0

    return im


def load(path):

    """ Loads, augment, and preprocess training images.
    :param path: The path to the training images folder
    :return: The tranformed images and labels
    """

    labelsList = os.listdir(path)
    data = []
    labels_list = []
    # Each sub-file corresponds to the label of the corresponding images
    for label in labelsList:
        temp = os.path.join(path, label, "*")
        # Number of augmentation steps
        for _ in range(0, 10):
            # Collect all images of one label
            for file in glob.glob(temp):
                image = cv2.imread(file)
                # Augment the image
                if _ != 0:
                    image = augmentation(image)
                # Preprocess image
                image = preprocessing(image)
                data.append(image)
                labels_list.append(label)

    data = np.array(data)
    data = data.reshape(data.shape[0], 48, 38, 1)

    return data, labels_list


def encode_feature(ytrain, ytest):

    """ Applies label encoder to the labels of images.
    :param ytrain: Labels of training images
    :param ytest: Labels of test images
    :return: The transformed labels and the LabelEncoder object
    """

    # Instantiate label encoder transform
    le_encoder = LabelEncoder()
    le = le_encoder.fit(ytrain)
    # Transform labels
    ytrain = le_encoder.transform(ytrain)
    ytest = le_encoder.transform(ytest)

    return ytrain, ytest, le


def build(input_shape):

    """ Builds network's architecture.
    :param input_shape: Input images dimensions
    :return: The model
    """

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.AvgPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.AvgPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.AvgPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(27, activation='softmax'))

    return model


def plot(train_metric, val_metric, low):

    """ Plots the learning curves of the training process.
    :param train_metric: The metric of the learning curve for the training set
    :param val_metric: The metric of the learning curve for the validation set
    :param low: Lower-bound of the y-axis of the plot
    """

    plt.figure(figsize=(7, 5))
    plt.plot(history.history[train_metric], label=train_metric)
    plt.plot(history.history[val_metric], label=val_metric)
    plt.xlabel('Epoch')
    plt.ylabel(train_metric)
    plt.ylim([low, 1])
    plt.legend(loc="best")
    plt.savefig("./src/training/" + train_metric + ".jpg")


if __name__ == '__main__':

    # Set path to the monkbrill data
    data_path = "./DSS/monkbrill/"

    # Load dataset and labels
    dataset, labels = load(data_path)

    # Split dataset to train and test set
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, shuffle=True, random_state=42)
    # Apply LabelEncoder() transformation to labels
    y_train, y_test, le = encode_feature(y_train, y_test)

    # Build the model
    cnn = build(dataset[0].shape)

    # Define optimizer and compile the model
    optim = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    cnn.compile(optimizer=optim,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model on train set
    history = cnn.fit(x_train, y_train, batch_size=64, epochs=25, validation_data=(x_test, y_test))
    # Evaluate its performance on test set
    test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=2)

    # Save trained model for later use
    cnn.save("./src/training/model.h5", include_optimizer=True)

    # Save learning curves
    plot('accuracy', 'val_accuracy', 0.5)
    plot('loss', 'val_loss', 0)

    # Save summary and label encoder transformation
    with open('./src/training/model.txt', 'w') as file:
        cnn.summary(print_fn=lambda x: file.write(x + '\n'))
    with open('./src/training/LabelEncoder.pickle', 'wb') as f:
        pickle.dump(le, f, pickle.HIGHEST_PROTOCOL)
