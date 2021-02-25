"""
Applies HoughLinesP to find lines in a prediction image.
NOT RECOMMENDED, just experimenting.
Check other methods in live.py with 'mode' parameter.
"""

from math import degrees, atan2

import cv2
import numpy as np

import config
from steering import steer
from utils import fna
from vector2line import vector2line


CLEAN_THRESHOLD = 3.5


def hough_lines_p(file_path: str, file_name: str):
    """ Reads the predicted lane image, applies HoughLinesP, and tries to steer """
    # read the predicted image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # model result is 256x128, it is still possible to reduce the size to 200x60
    img = cv2.resize(img, (200, 100))
    img = np.array(img)
    img = img[40 : 40 + 60, 0:200]

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )

    try:
        if lines is None or len(lines) < 2:
            print(
                f"{config.CC_WARNING}Cannot detect lines on the lanes{config.CC_ENDC}"
            )
            return
    except:
        pass

    cleans = np.empty(shape=[0, 4], dtype=np.int32)

    for line in lines:
        # degree between two points
        alpha = degrees(atan2(line[0][2] - line[0][0], line[0][3] - line[0][1]))

        if len(cleans) == 0:
            cleans = np.append(cleans, [line[0]], axis=0)
            continue

        similar = False
        for clean in cleans:
            beta = degrees(atan2(clean[2] - clean[0], clean[3] - clean[1]))
            if abs(alpha - beta) <= CLEAN_THRESHOLD:
                similar = True
                break

        if not similar:
            cleans = np.append(cleans, [line[0]], axis=0)

    lines = cleans
    direction, error = vector2line(lines, img, fna(file_name, "hlp"))
    if error == 0:
        steer(direction)
