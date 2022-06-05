""" This module implements the segmentation of the text images. """

import os
import cv2
import numpy as np
from findpeaks import findpeaks
from utils import makeDir, save_image


def contours_detection(img):

    """ Extract contours in input image that satisfy multiple conditions.
    :param img: The input image
    :return: A list with the coordinates of the detected contours
    """

    THIN_THRESHOLD = 12
    HEIGHT_THRESHOLD = 27

    # Find contours
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 3)

    # Delete child contours
    contours = list(contours)
    idx = []
    for i in range(hierarchy[0].shape[0]):
        # Third column in contours [2:3] specify the child contour if any
        if hierarchy[0][:, 2:3, ][i][0] != -1:
            idx.append(hierarchy[0][:, 2:3, ][i][0])

    for index in sorted(idx, reverse=True):
        del contours[index]
    contours = tuple(contours)

    # Extract coordinates of remained contours
    h_list = []
    total = 0
    dropout = 0

    # Find average width of contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < HEIGHT_THRESHOLD or w < THIN_THRESHOLD:
            dropout += 1
            continue
        total += w
    avg_width = total/(len(contours)-dropout)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Skip very small contours
        if h < HEIGHT_THRESHOLD or w < THIN_THRESHOLD or h+w < 55:
            continue

        # Separate multiple characters based on the average width
        if round(w/avg_width) >= 2:
            h_list.append([x, y, round(avg_width), h])
            h_list.append([x+w-round(avg_width), y, round(avg_width), h])
        if round(w/avg_width) == 1 or np.ceil(w/avg_width) == 1:
            h_list.append([x, y, w, h])

    h_list = sorted(h_list, key=lambda coord: (coord[1], coord[0]))[1:]

    return h_list


def detectValleys(image, index, lookahead) -> np.ndarray:

    """ Detect valleys on horizontal projection
    :param image: The input image
    :param index: The index of the projection, i.e., 0 for columns (vertical) and 1 for rows (horizontal)
    :param lookahead: Parameter of importance
    :return: The detected valleys
    """

    # Transform 2D matrix to 1D row/column vector
    histogram = cv2.reduce(image, index, cv2.REDUCE_AVG)

    # Find top peaks and valleys
    fp = findpeaks(lookahead=lookahead)
    results = fp.fit(histogram.reshape(-1))

    # Extract coordinates of valleys
    valleys = results['df'].query('valley == True')['y'].index.values
    valleys = valleys[1:]

    return valleys


def characters_segmentation(output_file, img, h_list, valleys):

    """ Segments input image into characters.
    :param output_file: The output file
    :param img: The input image
    :param h_list: The coordinates of the extracted contours
    :param valleys: The detected valleys
    """

    # Estimate cut-out lines
    cut = np.zeros(len(valleys)-1)
    for i in range(len(valleys)-1):
        cut[i] = int(0.3*valleys[i] + 0.7*valleys[i+1])

    # Find bottom right corners' y-coordinate of contours
    corners = [row[1] + row[3] for row in h_list]
    sorted_h_list = np.insert(h_list, 4, corners, axis=1)
    # Sort contours based on the y-coordinate of the bottom right corner (coord[4])
    sorted_h_list = sorted(list(sorted_h_list), key=lambda coord: (coord[4]))

    roi = []
    line_idx = 0
    for i in range(len(sorted_h_list)-1):
        # Collect contours that are above the current cut-out line
        if sorted_h_list[i][4] <= cut[line_idx]:
            x, y, w, h = sorted_h_list[i][0], sorted_h_list[i][1], sorted_h_list[i][2], sorted_h_list[i][3]
            roi.append([x, y, w, h])
        else:
            # When all contours of current cut-out line are collected, sort and save them
            order(img, roi, line_idx, output_file)
            roi = []
            # Collect the first contour of the new cut-out line and change line index
            x, y, w, h = sorted_h_list[i][0], sorted_h_list[i][1], sorted_h_list[i][2], sorted_h_list[i][3]
            roi.append([x, y, w, h])
            line_idx += 1
    # Sort and save the last line's contours
    order(img, roi, line_idx, output_file)


def order(img, roi, line_idx, output_file):

    """ Saves the contours in order.
    :param img: The input image
    :param roi: The coordinates of the region of interest
    :param line_idx: The current line index
    :param output_file: The output file
    """

    # Sort contours of a single line based on x-coordinate (coord[0])
    roi = sorted(list(roi), key=lambda coord: (coord[0]))
    for i in range(len(roi)):
        x, y, w, h = roi[i]
        # Extract the character inside the contour
        cnt = img[y:y + h, x:x + w]
        line_path = os.path.join(output_file, "Line_" + str(line_idx))
        makeDir(line_path)
        characters = os.path.join(line_path, "character_" + str(i))
        save_image(cnt, characters)
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)
    cv2.imwrite(output_file + '/final.png', img)


def segment(segment_output_file, image) -> list:

    """ Implements the segmentation steps.
    :param segment_output_file: The output file of segmentation
    :param image: The input image
    """

    # Find contours
    h_list = contours_detection(image)
    # Estimate cut-out lines
    valleys = detectValleys(image, 1, 60)
    # Segment characters
    characters_segmentation(segment_output_file, image, h_list, valleys)
