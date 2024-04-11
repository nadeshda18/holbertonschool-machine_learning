#!/usr/bin/env python3
"""add two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """add two matrices element-wise
    if mat1 and mat2 are not the same shape, return None
    Args:
        mat1: first matrix
        mat2: second matrix
    Returns:
        sum of two matrices"""
    """checks the number of rows//checks the number of columns"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    """inner loop for j in range(len(mat1[i]))over the columns of the matrix
    the outer loop for i in range(len(mat1))over the rows of the matrix"""
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
