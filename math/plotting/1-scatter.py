#!/usr/bin/env python3
"""plot x => y as a scatter graph"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """mean = [69, 0] mean values
    cov = [[15, 8], [8, 15]] covariance matrix
    np.random.seed(5) random number generator"""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    """scatter plot of x and y with magenta dots
    x-axis label = Height (in)
    y-axis label = Weight (lbs)
    title ="""
    plt.scatter(x, y, c='magenta', s=10)
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.show()
