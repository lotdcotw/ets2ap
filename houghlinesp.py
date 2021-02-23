"""
Applies HoughLinesP to find lines in a prediction image.
NOT for the solution, just experimenting
Check other methods in live.py with 'mode' parameter.
"""

from math import degrees, atan2

import cv2
import numpy as np

import config
from keys import right, left, straight
from utils import fna


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
    count = len(lines)

    print(f"{config.CC_HEADER}LINES{config.CC_ENDC}")
    print(lines)

    left_line = [0, 0, 0, 0]
    right_line = [0, 0, 0, 0]

    if count < 2:
        print(f"{config.CC_WARNING}Failed to detect at least two lines{config.CC_ENDC}")
        return

    # if more than 2 lines, selecting ones close to the camera center
    lines = sorted(lines, key=lambda x: abs(100 - (x[0] + x[2]) / 2))
    lines = [lines[0], lines[1]]

    if lines[0][0] < lines[1][0] and lines[0][2] < lines[1][2]:
        left_line = lines[0]
        right_line = lines[1]
    else:
        left_line = lines[1]
        right_line = lines[0]

    print(f"{config.CC_OKBLUE}Left Line: {config.CC_ENDC}", end="")
    print(left_line)
    print(f"{config.CC_OKBLUE}Right Line: {config.CC_ENDC}", end="")
    print(right_line)

    # DEBUG: draw lines on the image
    cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), 255, 2)
    cv2.line(
        img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), 255, 2
    )
    cv2.imwrite(fna(file_name, "hlp"), img)

    # middle points
    lxm = (left_line[0] + left_line[2]) / 2
    rxm = (right_line[0] + right_line[2]) / 2
    lym = (left_line[1] + left_line[2]) / 2
    rym = (right_line[1] + right_line[2]) / 2

    if lxm >= 100 and rxm >= 100 or lxm <= 100 and rxm <= 100:
        print(
            f"{config.CC_WARNING}Confused{config.CC_ENDC} while selectiong left/right lanes"
        )
    elif lym / rym > 1.5 or lym / rym < 0.5:
        print(f"{config.CC_WARNING}Confused{config.CC_ENDC} because of close lanes")
    else:
        midpoints = [
            [lxm, (left_line[1] + left_line[3]) / 2],
            [rxm, (right_line[1] + right_line[3]) / 2],
        ]
        center = [100, 60]  # bottom center of the image (original 200x60)
        diff_x = [abs(center[0] - midpoints[0][0]), abs(center[0] - midpoints[1][0])]
        direction = diff_x[0] / diff_x[1]
        print(f"{config.CC_OKBLUE}Diff X:{config.CC_ENDC} %s" % direction)
        print(f"{config.CC_OKCYAN}Steering:{config.CC_ENDC}", end="")

        if direction > 2:
            print(f"{config.CC_BOLD}<<<{config.CC_ENDC}")
            left()
        elif direction < 0.5:
            print(f"{config.CC_BOLD}>>>{config.CC_ENDC}")
            right()
        else:
            print(f"{config.CC_BOLD}^^^{config.CC_ENDC}")
            straight()
