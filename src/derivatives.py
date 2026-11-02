import numpy as np


def derivative_x(f, h):
    return (f[1:-1, 2:] - f[1:-1, :-2]) / (2 * h)


def derivative_y(f, h):
    return (f[2:, 1:-1] - f[:-2, 1:-1]) / (2 * h)


def second_derivative_x(f, h):
    return (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / h**2


def second_derivative_y(f, h):
    return (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / h**2


def laplacian(f, h):
    return second_derivative_x(f, h) + second_derivative_y(f, h)
