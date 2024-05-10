#!/usr/bin/env python3
"""returns two placeholders, x and y, for the neural network"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    """nx = number of feature columns in our data
    classes = number of classes in our classifier"""
    x = tf.placeholder("float", [None, nx], name="x")
    y = tf.placeholder("float", [None, classes], name="y")

    return x, y
