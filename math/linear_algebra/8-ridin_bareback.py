#!/usr/bin/env python3
"""performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """performs matrix multiplication
    Args:
        mat1: first matrix
        mat2: second matrix
    Returns:
        product of two matrices"""
    """checks if number of columns in mat1
    is equal to number of rows in mat2"""
    """a=mat1_row, b=mat2_col"""
    """zip(*mat2) is transposing mat2"""
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a * b for a, b in zip(mat1_row, mat2_col))
             for mat2_col in zip(*mat2)]
            for mat1_row in mat1]
