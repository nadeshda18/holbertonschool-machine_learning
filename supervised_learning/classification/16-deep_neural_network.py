#!/usr/bin/env python3
"""defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """class DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """nx is the number of input features
        layers is a list representing the number of nodes in each
        layer of the network
        L: The number of layers in the neural network.
        cache: A dictionary to hold all intermediary values of the network.
        weights: A dictionary to hold all weights and biased of the network.
        The weights of the network should be initialized using the He et al.
        method and the biases should be initialized to 0's"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")
        # attribute L = number of layers in the neural network
        self.L = len(layers)
        # cache = empty dictionary to hold all intermediary values of the network
        self.cache = {}
        # weights = empty dictionary to hold all weights and biased of the network
        self.weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            # if layer is the first layer
            if i == 0:
                # He et al. method to initialize weights
                self.weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
                # bias initialized to 0
                self.weights['b1'] = np.zeros((layers[i], 1))
            else:
                # He et al. method to initialize weights if i > 0
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
                # bias initialized to 0
                self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
