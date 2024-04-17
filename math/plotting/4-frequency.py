#!/usr/bin/env python3
"""plot a histogram of student
scores for a project"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """np.random.seed(5) random number generator
    student_grades = random"""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xticks(range(0, 101, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.show()
