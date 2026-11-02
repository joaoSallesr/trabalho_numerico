import numpy as np


def create_grid(N, xlim=(-1, 1), ylim=(-1, 1)):
    x = np.linspace(xlim[0], xlim[1], N + 1)
    y = np.linspace(ylim[0], ylim[1], N + 1)
    h = (xlim[1] - xlim[0]) / N

    X, Y = np.meshgrid(x, y)

    return x, y, X, Y, h
