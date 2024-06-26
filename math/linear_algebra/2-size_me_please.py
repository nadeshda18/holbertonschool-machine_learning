#!/usr/bin/env python3
"""calculate the shape of a matrix"""


def matrix_shape(matrix):
    """all elements are of same type/shape
    Args:
        matrix: list of elements"""
    if type(matrix) is not list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
