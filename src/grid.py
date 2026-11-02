import numpy as np


def create_grid(N):
    x = np.linspace(-1, 1, N + 1)
    y = np.linspace(-1, 1, N + 1)
    h = 2 / N

    X, Y = np.meshgrid(x, y)

    return x, y, X, Y, h
