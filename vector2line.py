"""
Tries to find lines in the given vectors.
"""

import cv2

import config
from utils import fna


def vector2line(lines, img=None, file_name: str = ""):
    """ Tries to find appropriate lanes in a vector array """

    count = len(lines)

    print(f"{config.CC_HEADER}LINES{config.CC_ENDC}")

    left_line = [0, 0, 0, 0]
    right_line = [0, 0, 0, 0]

    if count < 2:
        print(f"{config.CC_WARNING}Failed to detect at least two lines{config.CC_ENDC}")
        return (0, 1)

    print(lines)

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

    # draw lines on the image
    if img is not None and file_name != "":
        cv2.line(
            img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), 255, 2
        )
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
        return (0, 2)
    elif lym / rym > 1.5 or lym / rym < 0.5:
        print(f"{config.CC_WARNING}Confused{config.CC_ENDC} because of close lanes")
        return (0, 3)

    midpoints = [
        [lxm, (left_line[1] + left_line[3]) / 2],
        [rxm, (right_line[1] + right_line[3]) / 2],
    ]
    center = [100, 60]  # bottom center of the image (original 200x60)
    diff_x = [abs(center[0] - midpoints[0][0]), abs(center[0] - midpoints[1][0])]
    direction = diff_x[0] / diff_x[1]
    print(f"{config.CC_OKBLUE}Diff X:{config.CC_ENDC} %s" % direction)

    return (direction, 0)
