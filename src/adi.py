import numpy as np


def coefficients_x(u, h, Re):
    u = u[1:-1, 1:-1]

    a_W = 1 / (Re * h**2) + u / (2 * h)
    a_P = np.full_like(u, -2 / (Re * h**2))
    a_E = 1 / (Re * h**2) - u / (2 * h)

    return a_W, a_P, a_E


def coefficients_y(v, h, Re):
    v = v[1:-1, 1:-1]

    b_S = 1 / (Re * h**2) + v / (2 * h)
    b_P = np.full_like(v, -2 / (Re * h**2))
    b_N = 1 / (Re * h**2) - v / (2 * h)

    return b_S, b_P, b_N


def adi_first_step():
    pass


def adi_second_step():
    pass
