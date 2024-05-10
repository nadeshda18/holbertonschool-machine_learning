#!/usr/bin/env python3
"""creates a layer, returns tensor output of the layer"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """prev = tensor output of the previous layer
    n = number of nodes in the layer to create
    activation = activation function that the layer should use"""
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=initializer, name="layer")
    return layer(prev)
