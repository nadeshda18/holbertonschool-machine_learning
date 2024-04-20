#!/usr/bin/env python3
"""calculate the sum"""


def summation_i_squared(n):
    """calculate the sum"""
    if type(n) is not int:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
