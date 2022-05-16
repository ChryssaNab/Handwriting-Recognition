import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models


def preprocessing(img):
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
    labels = []
    for label in labelsList:
        temp = os.path.join(path, label, "*")
        for file in glob.glob(temp):
            image = cv2.imread(file)
            # Preprocess image
            image = preprocessing(image)
            data.append(image)
            labels.append(label)

    data = np.array(data)
    data = data.reshape(data.shape[0], 48, 38, 1)

    return data, labels


def encode_feature(ytrain, ytest):
    le_encoder = LabelEncoder()
    le_encoder.fit(ytrain)

    ytrain = le_encoder.transform(ytrain)
    ytest = le_encoder.transform(ytest)

    return ytrain, ytest, le_encoder


def build(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(27))

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
    plt.show()


if __name__ == '__main__':

    # Set path to monkbrill data
    data_path = "" 

    # Load dataset
    dataset, labels = load(data_path)
    # Split dataset to train, test and validation set
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, shuffle=True, random_state=42)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    # Apply LabelEncoder() transformation to labels
    y_train, y_test, le = encode_feature(y_train, y_test)

    # Build the model
    cnn = build(dataset[0].shape)

    # Define optimizer and compile the model
    optim = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    cnn.compile(optimizer=optim,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model
    history = cnn.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test))
    test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=2)

    # Save plots
    # plot('accuracy', 'val_accuracy', 0.5)
    # plot('loss', 'val_loss', 0)

    # Save pre-trained model
    cnn.save("model.h5", include_optimizer=True)

    # Save summary
    with open('./model.txt', 'w') as file:
        cnn.summary(print_fn=lambda x: file.write(x + '\n'))
