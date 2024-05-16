#!/usr/bin/env python3
"""creates mini-batches to be used for training a neural network
using mini-batch gradient descent"""
import numpy as np


def create_mini_batches(X, Y, batch_size):
    """X = numpy.ndarray of shape (m, nx) input data
    m = number of data points
    nx = number of features in X
    Y = numpy.ndarray of shape (m, ny) the labels
    m = same number of data points as in X
    ny = number of classes for classification tasks
    batch_size = data points in a batch
    Returns: a list mini-batches of tuples (X_batch, Y_batch)"""
    m = X.shape[0]
    mini_batches = []

    shuffle_data = __import__('2-shuffle_data').shuffle_data
    X, Y = shuffle_data(X, Y)

    num_complete_minibatches = m // batch_size

    for i in range(0, m, batch_size):
        X_batch = X[i * batch_size: i * batch_size + batch_size, :]
        Y_batch = Y[i * batch_size: i * batch_size + batch_size, :]
        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        X_batch = X[num_complete_minibatches * batch_size: m, :]
        Y_batch = Y[num_complete_minibatches * batch_size: m, :]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
