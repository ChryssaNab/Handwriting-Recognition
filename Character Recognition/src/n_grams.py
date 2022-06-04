""" This module implements the bigram language model of Hebrew characters. """

import pandas as pd
import numpy as np


def getProbs(path):

    """ Extracts the probabilities of bigrams.
    :param path: The path to the frequencies file of Hebrew characters
    :return: The probabilities of bigrams in the form of array
    """

    data = pd.read_csv(path)
    probability_matrix = np.zeros((27, 27))
    labels = ['Alef', 'Bet', 'Gimel', 'Dalet', 'He', 'Waw', 'Zayin', 'Het', 'Tet', 'Yod','Kaf','FinalKaf','Lamed','Mem-medial','Mem-final','Nun-medial','Nun-final','Samekh','Ayin','Pe-medial','Pe-final','Tsadi','Qof','Resh','Shin','Taw']

    for m in range(0, data['Names'].size):
        count = data['Names'][m].split('_')
        for i in range(0, len(count)):
            for j in range(0, len(labels)):
                if count[i] == labels[j] and i > 0:
                    probability_matrix[prev][j] = probability_matrix[prev][j] + data['Frequencies'][m]
                    prev = j
                else:
                    if count[i] == labels[j]:
                        prev = j

    # Normalize probabilities
    for i in range(0, len(labels)):
        probability_matrix[i] = probability_matrix[i] / sum(probability_matrix[i])
    return probability_matrix

