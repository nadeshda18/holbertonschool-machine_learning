#!/usr/bin/env python3
"""creates the training operation for the network"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """loss = loss of the network's prediction
    alpha = learning rate
    Returns: an operation that trains the network using
    gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
