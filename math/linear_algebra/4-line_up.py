#!/usr/bin/env python3
"""add two arrays element-wise"""


def add_arrays(arr1, arr2):
    """add two arrays element-wise
    if arr1 and arr2 are not the same shape, return None
    Args:
        arr1: first array
        arr2: second array
    Returns:
        sum of two arrays"""
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
