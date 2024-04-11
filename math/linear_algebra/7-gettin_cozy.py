#!/usr/bin/env python3
"""concatenate two matrices
along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenate two matrices
    along a specific axis
    Args:
        mat1: first matrix
        mat2: second matrix
        axis: axis to concatenate
    Returns:
        concatenated matrix"""
    """axis=0 concatenates the matrices along the rows"""
    """axis=1 concatenates the matrices along the columns"""
    """checks the number of rows"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]
    """checks the number of columns"""
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i].copy() + mat2[i].copy() for i in range(len(mat1))]
