""" This module implements the character recognition. """

import os
import cv2
import pickle
import numpy as np
from natsort import natsorted
from transcript import script

with open('./training/LabelEncoder.pickle', 'rb') as f:
    LabelEncoder = pickle.load(f)


def network_prediction(character_img, model):

    """ Recognizes single characters.
    :param character_img: The character to be recognized
    :param model: The trained model
    :return: The name of the predicted character
    """

    # Reshape character to fit network's input layer
    character_img = cv2.resize(character_img, (38, 48), interpolation=cv2.INTER_LANCZOS4)
    character = np.reshape(character_img, (1, 48, 38, 1))
    # Predict character
    predicted = model.predict(character)
    # Get the name of the predicted character
    character = LabelEncoder.inverse_transform([np.argmax(predicted)])

    return character


def predict(output_dir, transcript_output, model, n_grams):

    """ Implements character recognition.
    :param output_dir: Folder with segmented characters
    :param transcript_output: Folder with output transcription
    :param model: The trained model for the recognition
    :param n_grams:
    """

    # Collect all lines of one image
    lines = [line for line in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, line))]
    # Sort lines
    lines = natsorted(lines)

    # Create file for transcription
    file = os.path.join(transcript_output, os.path.basename(output_dir) + ".txt")
    f_txt = open(file, "w", encoding="utf-8")

    for line in lines:
        # Load single line
        line_path = os.path.join(output_dir, line)
        # Collect all characters of single line
        characters = os.listdir(line_path)
        # Sort characters
        characters = natsorted(characters)

        line_list = []
        for character in characters:
            # Load single character
            character_path = os.path.join(line_path, character)
            character_img = cv2.imread(character_path, cv2.IMREAD_UNCHANGED)
            # Recognize character using the trained model
            cnn_character = network_prediction(character_img, model)
            # Append all recognized characters of one line to a list
            line_list.append(cnn_character[0])

        # Implement transcription for each single line
        f_txt = script(f_txt, line_list)
    f.close()

