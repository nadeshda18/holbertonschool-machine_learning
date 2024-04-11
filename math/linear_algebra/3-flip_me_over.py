#!/usr/bin/env python3
"""transpose a 2D matrix"""


def matrix_transpose(matrix):
    """transpose a 2D matrix
    Args:
        matrix: 2D matrix
    Returns: new matrix
    """
    """inner loop iterates for j in range(len(matrix)over the rows of the
    matrix
    and appends the elements of the rows to the columns of the new matrix
    outer loop iterates for i in range(len(matrix)over the columns of the
    matrix
    and appends the elements"""
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
