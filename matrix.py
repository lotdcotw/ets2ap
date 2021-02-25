"""
Finds and process the lines in a matrix tensor[0][0]
"""

from matrix2vector import find_vectors_in_matrix_data
from steering import steer
from vector2line import vector2line


def matrix(data):
    """ Tries to find lines in a matrix, and predict steering direction and sensitivity """
    lines = find_vectors_in_matrix_data(data)
    direction, error = vector2line(lines)
    if error == 0:
        steer(direction)
