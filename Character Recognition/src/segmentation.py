import os
import cv2
import numpy as np
from findpeaks import findpeaks
from utils import makeDir, save_image


def contours_detection(img):
    THIN_THRESHOLD = 25

    # Find contours
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 3)

    # Delete child contours
    contours = list(contours)
    idx = []
    for i in range(hierarchy[0].shape[0]):
        # Third column in contours specify the child contour if any
        if hierarchy[0][:, 2:3, ][i][0] != -1:
            idx.append(hierarchy[0][:, 2:3, ][i][0])

    for index in sorted(idx, reverse=True):
        del contours[index]
    contours = tuple(contours)
    
    # Extract coordinates of remained contours
    h_list = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < THIN_THRESHOLD or w < THIN_THRESHOLD:
            continue
        h_list.append([x, y, w, h])

    h_list = sorted(h_list, key=lambda coord: (coord[1], coord[0]))[1:]

    return h_list


def detectValleys(image, index, lookahead) -> np.ndarray:
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
    # Define cut-out lines
    cut = np.zeros(len(valleys)-1)
    for i in range(len(valleys)-1):
        cut[i] = int(0.25*valleys[i] + 0.75*valleys[i+1])

    # Find bottom right corners' y coordinates
    corners = [row[1] + row[3] for row in h_list]

    sorted_h_list = np.insert(h_list, 4, corners, axis=1)
    sorted_h_list = sorted(list(sorted_h_list), key=lambda coord: (coord[4]))

    roi = []
    line_idx = 0
    for i in range(len(sorted_h_list)):
        if sorted_h_list[i][4] <= cut[line_idx]:
            x, y, w, h = sorted_h_list[i][0], sorted_h_list[i][1], sorted_h_list[i][2], sorted_h_list[i][3]
            roi.append([x, y, w, h])
        else:
            order(img, roi, line_idx, output_file)
            roi = []
            x, y, w, h = sorted_h_list[i][0], sorted_h_list[i][1], sorted_h_list[i][2], sorted_h_list[i][3]
            roi.append([x, y, w, h])
            line_idx += 1
    order(img, roi, line_idx, output_file)
    return roi
   
    
def order(img, roi, line_idx, output_file):
    roi = sorted(list(roi), key=lambda coord: (coord[0]))
    for i in range(len(roi)):
        x, y, w, h = roi[i]
        cnt = img[y:y + h, x:x + w]
        line_path = os.path.join(output_file, "Line_" + str(line_idx))
        makeDir(line_path)
        characters = os.path.join(line_path, "character_" + str(i))
        save_image(cnt, characters)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    cv2.imwrite(output_file + '/final.png', img)


def segment(segment_output_file, image) -> list:
    # Find contours
    h_list = contours_detection(image)
    # Find cut-out lines
    valleys = detectValleys(image, 1, 60)
    roi = characters_segmentation(segment_output_file, image, h_list, valleys)

    return roi
