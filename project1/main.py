"""
Starter code for EECS 442 W20 HW1
"""
from util import *
import numpy as np
import matplotlib.pyplot as plt


def rotX(theta):
    # TODO: Return the rotation matrix for angle theta around X axis
    pass


def rotY(theta):
    # TODO: Return the rotation matrix for angle theta around Y axis
    pass


def projectOthographicLines(R, t, L):
    pass


def part1():
    pass


def split_triptych(trip):
    # TODO: Split a triptych into thirds and return three channels in numpy arrays
    pass


def normalized_cross_correlation(ch1, ch2):
    # TODO: Implement the default similarity metric
    pass


def best_offset(ch1, ch2, metric, Xrange=np.arange(-5, 16), Yrange=np.arange(-5, 16)):
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets
    pass


def align_and_combine(R, G, B, metric):
    # TODO: Use metric to align three channels and return the combined RGB image
    pass


def part2():
    pass


def part3():
    pass


def main():
    part1()
    part2()
    part3()


if __name__ == "__main__":
    main()
