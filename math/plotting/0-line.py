#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""plot y as a line graph"""


def line():
    """y as a red line"""
    """x should range from 0 - 10"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()
