#!/usr/bin/env python3
"""concatenates two matrices
along a specific axis"""
import numpy as np

def np_cat(mat1, mat2, axis=0):
    """concatenates two matrices
    along a specific axis
    Args:
        mat1: numpy.ndarray
        mat2: numpy.ndarray
        axis: axis to concatenate
    Returns:
        concatenated matrix"""
    """for axis=0, along the rows, stacked one
    on top of the other/number of columns must
    be the same"""
    """for axis=1, along the columns, stacked side
    by side/number of rows must be the same"""
    return np.concatenate((mat1, mat2), axis=axis)
