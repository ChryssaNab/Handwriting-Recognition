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

np_config.enable_numpy_behavior()
from dataAugmentation import *




def preprocessing(img):
    #Augment the image
    img = dataAugmentation.randomaugmentimage(img)
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
    labelsList = os.listdir(path)
    data = []
    labels_list = []
    for label in labelsList:
        temp = os.path.join(path, label, "*")
        for _ in range(0, 10):
            for file in glob.glob(temp):
                image = cv2.imread(file)
                # Preprocess image
                image = preprocessing(image)
                data.append(image)
                labels_list.append(label)

    data = np.array(data)
    data = data.reshape(data.shape[0], 48, 38, 1)

    return data, labels_list


def encode_feature(ytrain, ytest):
    le_encoder = LabelEncoder()
    le = le_encoder.fit(ytrain)

    ytrain = le_encoder.transform(ytrain)
    ytest = le_encoder.transform(ytest)

    return ytrain, ytest, le


def buildLeNet(input_shape):
    model = keras.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.AveragePooling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='relu'))

    model.add(layers.Dense(units=84, activation='relu'))

    model.add(layers.Dense(units=27, activation='softmax'))
    return model

def build(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.AvgPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.AvgPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.AvgPool2D((2, 2)))
    #model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    #model.add(layers.AvgPool2D((2, 2)))
    model.add(layers.Dense(128, activation='softmax'))
    model.add(layers.Dense(64, activation='softmax'))
    model.add(layers.Flatten())
    model.add(layers.Dense(27, activation='softmax'))

    return model


def plot(train_metric, val_metric, low):
    plt.figure(figsize=(7, 5))
    plt.plot(history.history[train_metric], label=train_metric)
    plt.plot(history.history[val_metric], label=val_metric)
    plt.xlabel('Epoch')
    plt.ylabel(train_metric)
    plt.ylim([low, 1])
    plt.legend(loc="best")
    plt.savefig(train_metric + ".jpg")


if __name__ == '__main__':
    data_path = "/home/chryssa/Desktop/Groningen/RUG/Semester 2b/Handwriting Recognition/Assignment/Methods/Task1/Data/DDS/monkbrill"

    # Load dataset
    dataset, labels = load(data_path)

    # Split dataset to train, test and validation set
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, shuffle=True, random_state=42)
    # Apply LabelEncoder() transformation to labels
    y_train, y_test, le = encode_feature(y_train, y_test)

    # Build the model
    cnn = build(dataset[0].shape)
    #cnn = buildLeNet(dataset[0].shape)

    # Define optimizer and compile the model
    optim = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    cnn.compile(optimizer=optim,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model
    history = cnn.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_test, y_test))
    # history = cnn.fit_generator(datagen.flow(x_train, y_train, batch_size=64), validation_data=(x_test, y_test), epochs=80)

    test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=2)

    # Save pre-trained model
    cnn.save("model.h5", include_optimizer=True)

    # Save plots
    plot('accuracy', 'val_accuracy', 0.5)
    plot('loss', 'val_loss', 0)


    # Save summary
    with open('./model.txt', 'w') as file:
        cnn.summary(print_fn=lambda x: file.write(x + '\n'))

    with open('LabelEncoder.pickle', 'wb') as f:
        pickle.dump(le, f, pickle.HIGHEST_PROTOCOL)

