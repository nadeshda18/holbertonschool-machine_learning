#!/usr/bin/env python3
"""perform element-wise add/sub/
mult/div on mat1 and mat2"""


def np_elementwise(mat1, mat2):
    """perform element-wise add/sub/
    mult/div on mat1 and mat2
    Args:
        mat1: first matrix
        mat2: second matrix
    Returns:
        tuple containing the element-wise
        sum, difference, product, and quotient"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
