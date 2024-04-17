#!/usr/bin/env python3
"""plot y as a line graph"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """y as a red line"""
    """x should range from 0 - 10"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()
