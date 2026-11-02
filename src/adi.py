import numpy as np
from scipy.linalg import solve_banded

from .exact_solutions import omega_exact


def coefficients_x(u, h, Re):
    u_int = u[1:-1, 1:-1]

    a_W = 1 / (Re * h**2) + u_int / (2 * h)
    a_P = np.full_like(u_int, -2 / (Re * h**2))
    a_E = 1 / (Re * h**2) - u_int / (2 * h)

    return a_W, a_P, a_E


def coefficients_y(v, h, Re):
    v_int = v[1:-1, 1:-1]

    b_S = 1 / (Re * h**2) + v_int / (2 * h)
    b_P = np.full_like(v_int, -2 / (Re * h**2))
    b_N = 1 / (Re * h**2) - v_int / (2 * h)

    return b_S, b_P, b_N


def apply_bc(omega, X, Y, t, Re):

    omega[0, :] = omega_exact(X[0, :], Y[0, :], t, Re)
    omega[-1, :] = omega_exact(X[-1, :], Y[-1, :], t, Re)
    omega[:, 0] = omega_exact(X[:, 0], Y[:, 0], t, Re)
    omega[:, -1] = omega_exact(X[:, -1], Y[:, -1], t, Re)

    return omega


# Algoritmo de Thomas
def tdma(lower, main, upper, rhs):
    n = len(main)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper  # linha 0: super-diagonal
    ab[1, :] = main  # linha 1: diagonal principal
    ab[2, :-1] = lower  # linha 2: sub-diagonal
    return solve_banded((1, 1), ab, rhs)


# Passo ADI de Peaceman-Rachford
def adi_step(omega, u, v, dt, h, Re, t, X, Y):
    N = omega.shape[0] - 1  # número de subintervalos (N+1 nós por direção)
    Ni = N - 1  # número de nós interiores por direção
    c = dt / 2.0  # comprimento do meio-passo

    # Coeficientes dos operadores Lx e Ly
    a_W, a_P, a_E = coefficients_x(u, h, Re)
    b_S, b_P, b_N = coefficients_y(v, h, Re)

    # ===================================== STEP 1 =====================================
    omega_half = omega.copy()
    apply_bc(omega_half, X, Y, t + c, Re)

    Ly = (
        b_S * omega[:-2, 1:-1]  # vizinho sul    ω[i-1, j]
        + b_P * omega[1:-1, 1:-1]  # centro      ω[i,   j]
        + b_N * omega[2:, 1:-1]  # vizinho norte ω[i+1, j]
    )

    rhs_half = omega[1:-1, 1:-1] + c * Ly

    for i in range(Ni):
        rhs = rhs_half[i].copy()

        # Incorpora Condição de Contorno
        rhs[0] += c * a_W[i, 0] * omega_half[i + 1, 0]
        rhs[-1] += c * a_E[i, -1] * omega_half[i + 1, -1]

        # Diagonais de (I - c*Lx)
        lower = -c * a_W[i, 1:]  # sub-diagonal   (tamanho Ni-1)
        main = 1 - c * a_P[i, :]  # diagonal ppal  (tamanho Ni)
        upper = -c * a_E[i, :-1]  # super-diagonal (tamanho Ni-1)

        omega_half[i + 1, 1:-1] = tdma(lower, main, upper, rhs)

    # ===================================== STEP 2 =====================================
    omega_new = omega_half.copy()
    apply_bc(omega_new, X, Y, t + dt, Re)

    Lx = (
        a_W * omega_half[1:-1, :-2]  # vizinho oeste   ω*[i, j-1]
        + a_P * omega_half[1:-1, 1:-1]  # centro       ω*[i,   j]
        + a_E * omega_half[1:-1, 2:]  # vizinho leste  ω*[i, j+1]
    )

    rhs_new = omega_half[1:-1, 1:-1] + c * Lx

    for j in range(Ni):
        rhs = rhs_new[:, j].copy()

        # Incorpora Condição de Contorno
        rhs[0] += c * b_S[0, j] * omega_new[0, j + 1]
        rhs[-1] += c * b_N[-1, j] * omega_new[-1, j + 1]

        # Diagonais de (I - c*Ly)
        lower = -c * b_S[1:, j]  # sub-diagonal   (tamanho Ni-1)
        main = 1 - c * b_P[:, j]  # diagonal ppal  (tamanho Ni)
        upper = -c * b_N[:-1, j]  # super-diagonal (tamanho Ni-1)

        omega_new[1:-1, j + 1] = tdma(lower, main, upper, rhs)

    return omega_new
