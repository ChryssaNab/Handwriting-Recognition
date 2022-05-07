import os
import cv2
import numpy as np
from utils import save_image, makeDir
from findpeaks import findpeaks


def detectPeaks(image, index, lookahead) -> np.ndarray:
    # Transform 2D matrix to 1D row/column vector
    histogram = cv2.reduce(image, index, cv2.REDUCE_AVG)

    # Find top peaks 
    fp = findpeaks(lookahead=lookahead)
    results = fp.fit(histogram.reshape(-1))

    # Extract coordinates of peaks
    peaks = results['df'].query('peak == True')['y'].index.values

    return peaks


def lineSegmentation(lines_path, image, line_peaks) -> list:

    lines = []

    # Define first line
    start = 0
    end = line_peaks[0]
    lines.append(image[start:end])
    
    # Save first line in device
    p = os.path.join(lines_path, "Line_" + str(0))
    save_image(lines[0], p)

    for i in range(1, line_peaks.shape[0]):
        start = line_peaks[i-1]
        end = line_peaks[i]
        lines.append(image[start:end])

        # Save each segmented line in device
        p = os.path.join(lines_path, "Line_" + str(i))
        save_image(lines[i], p)

    return lines


def wordSegmentation(words_path, line, word_peaks) -> list:

    words = []
    
    # Define first word
    start = 0
    end = word_peaks[0]
    words.append(line[:, start:end])

    # Save first segmented word in device
    p = os.path.join(words_path, "word_" + str(0))
    save_image(words[0], p)

    for i in range(1, word_peaks.shape[0]):
        start = word_peaks[i-1]
        end = word_peaks[i]
        words.append(line[:, start:end])

        # Save each segmented word in device
        p = os.path.join(words_path, "word_" + str(i))
        save_image(words[i], p)

    return words


def segment(output_file, image) -> list:

    # Find higher peaks that correspond to lines
    line_peaks = detectPeaks(image, 1, 50)
    # Segment image in lines
    lines = lineSegmentation(output_file, image, line_peaks)

    lineIdx = 0
    for line in lines:
    
        words_path = os.path.join(output_file, "Words_L" + str(lineIdx))
        makeDir(words_path)
        
        # Find higher peaks that correspond to columns
        word_peaks = detectPeaks(line, 0, 35)
        if word_peaks.any():
            # Segment each line in words
            words = wordSegmentation(words_path, line, word_peaks)

        lineIdx += 1

    return words
