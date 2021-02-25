"""
Finds and process the lines in a matrix tensor[0][0]
"""

import cv2

from matrix2vector import find_vectors_in_matrix_data
from steering import steer
from vector2line import vector2line


def matrix(data, graph: bool = False, file_path: str = "", file_name: str = ""):
    """ Tries to find lines in a matrix, and predict steering direction and sensitivity """

    lines = find_vectors_in_matrix_data(data, graph, file_name)

    if graph:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        direction, error = vector2line(lines, img, file_name)
    else:
        direction, error = vector2line(lines)

    if error == 0:
        steer(direction)
