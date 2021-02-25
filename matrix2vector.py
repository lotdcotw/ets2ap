"""
Matrix and vector operations.
Matrix is an element of a tensor e.g. tensor[0][0]
"""

import argparse

import numpy as np
from scipy import spatial
from scipy.ndimage.measurements import label

import config
from utils import fna


THRESHOLD = 100


def find_vectors_in_matrix_file(
    matrix_file: str, cols: int, rows: int, graph: bool = False, file_name: str = ""
):
    """ Finds and returns vectors in a matrix file """
    data = []
    with open(matrix_file, "r") as f:
        data = [[int(num) for num in line.split(" ")] for line in f]

    if len(data) == 0:
        print(f"{config.CC_ERROR}Invalid matrix{config.CC_ENDC}")
        return []
    if len(data) != rows:
        print(f"{config.CC_ERROR}Invalid Y size{config.CC_ENDC}")
        return []
    elif len(data[0]) != cols:
        print(f"{config.CC_ERROR}Invalid X size{config.CC_ENDC}")
        return []
    return find_vectors_in_matrix(data, graph, file_name)


def find_vectors_in_matrix_data(data, graph: bool = False, file_name: str = ""):
    """
    Finds and returns vectors in a 2D matrix.
    This is usually the first element of a tensor e.g. tensor[0][0]
    """
    return find_vectors_in_matrix(data, graph, file_name)


def find_vectors_in_matrix(data, graph: bool = False, file_name: str = ""):
    """ Finds and returns vectors in a 2D matrix """

    labeled_array, num_features = label(data)  # group and label

    if graph:
        np.savetxt(fna(file_name, "matrix_labeled", "txt"), labeled_array, fmt="%d")

    vectors = []
    for i in range(num_features):
        index = i + 1
        indices = np.argwhere(labeled_array == index)  # find the group
        if len(indices) < THRESHOLD:  # not an important line if under threshold
            continue
        candidates = indices[spatial.ConvexHull(indices).vertices]  # convex hull
        dist_mat = spatial.distance_matrix(candidates, candidates)  # distance btw
        points = np.unravel_index(dist_mat.argmax(), dist_mat.shape)  # furthest cands
        vectors.append(
            [
                candidates[points[0]][0],
                candidates[points[0]][1],
                candidates[points[1]][0],
                candidates[points[1]][1],
            ]
        )

    return np.array(vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix2Vector")
    parser.add_argument(
        "--file", type=str, required=True, help="Path of the matrix file"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=256,
        help="Number of columns in a matrix",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=128,
        help="Number of rows in a matrix",
    )
    args = parser.parse_args()

    find_vectors_in_matrix_file(args.file, args.cols, args.rows)
