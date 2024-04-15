#!/usr/bin/env python3
"""performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """performs matrix multiplication
    Args:
        mat1: first matrix
        mat2: second matrix
    Returns:
        product of two matrices"""
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(len(mat1[0])):
                sum += mat1[i][k] * mat2[k][j]
            row_result.append(sum)
        result.append(row_result)
    return result
