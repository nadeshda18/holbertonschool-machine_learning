#!/usr/bin/env python3
"""calculates the softmax cross-entropy loss of a prediction"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """y = placeholder for the labels of the input data
    y_pred = tensor containing the network's predictions
    Returns: a tensor containing the loss of the prediction"""
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))
    return loss
