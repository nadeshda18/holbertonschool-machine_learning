#!/usr/bin/env python3
"""plot x => y as a line graph"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """x = range from 0 to 28650 by 5730
    r = log(0.5) rate of decay
    t = 5730 half-life of C-14
    y = exp(r/t) * x exponential decay of C-14"""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    """yscale = logarithmic scale
    x-axis range from 0-28650"""
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')
    plt.yscale('log')
    plt.xlim(0, 28650)
    plt.plot(x, y)
    plt.show()
