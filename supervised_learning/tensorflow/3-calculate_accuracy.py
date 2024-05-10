#!/usr/bin/env python3
"""calculates the accuracy of a prediction"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """y = placeholder for the labels of the input data
    y_pred = tensor containing the network's predictions
    Returns: a tensor containing the decimal accuracy of
    the prediction"""
    prediction = tf.argmax(y_pred, 1)
    correct = tf.argmax(y, 1)
    correct_prediction = tf.equal(prediction, correct)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
