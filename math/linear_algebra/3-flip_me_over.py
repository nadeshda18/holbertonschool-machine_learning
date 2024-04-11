#!/usr/bin/env python3
"""transpose a 2D matrix"""


def matrix_transpose(matrix):
    """transpose a 2D matrix
    Args:
        matrix: 2D matrix
    Returns: new matrix
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
