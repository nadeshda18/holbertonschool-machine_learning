#!/usr/bin/env python3
"""calculates the normalization (standardization) constants
of a matrix"""
import numpy as np


def normalization_constant(X):
    """X = numpy.ndarray of shape (m, nx) to normalize
    m = number of data points
    nx = number of features
    Returns: the mean and standard deviation of each feature"""
    # calculate the mean and standard deviation of each feature
    mean = np.mean(X, axis=0)

    std = np.std(X, axis=0)

    return mean, std
