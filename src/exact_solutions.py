import numpy as np


def omega_exact(X, Y, t, Re):
    return 2 * np.pi * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp((-2 * np.pi**2 * t) / Re)


def u_exact(X, Y, t, Re):
    return -np.cos(np.pi * X) * np.sin(np.pi * Y) * np.exp((-2 * np.pi**2 * t) / Re)


def v_exact(X, Y, t, Re):
    return np.sin(np.pi * X) * np.cos(np.pi * Y) * np.exp((-2 * np.pi**2 * t) / Re)
