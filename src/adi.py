import numpy as np


def coefficients_x(u, h, nu):
    u_int = u[1:-1, 1:-1]
    a_W = nu / h**2 + u_int / (2 * h)
    a_P = np.full_like(u_int, -2 * nu / h**2)
    a_E = nu / h**2 - u_int / (2 * h)
    return a_W, a_P, a_E


def coefficients_y(v, h, nu):
    v_int = v[1:-1, 1:-1]
    b_S = nu / h**2 + v_int / (2 * h)
    b_P = np.full_like(v_int, -2 * nu / h**2)
    b_N = nu / h**2 - v_int / (2 * h)
    return b_S, b_P, b_N


def thomas(lower, main, upper, rhs):
    n = main.shape[-1]
    cp = np.zeros_like(main)
    dp = np.zeros_like(rhs)

    # varredura para frente
    if n > 1:
        cp[:, 0] = upper[:, 0] / main[:, 0]
    dp[:, 0] = rhs[:, 0] / main[:, 0]
    for k in range(1, n):
        denom = main[:, k] - lower[:, k - 1] * cp[:, k - 1]
        if k < n - 1:
            cp[:, k] = upper[:, k] / denom
        dp[:, k] = (rhs[:, k] - lower[:, k - 1] * dp[:, k - 1]) / denom

    # substituição de volta
    x = np.zeros_like(rhs)
    x[:, -1] = dp[:, -1]
    for k in range(n - 2, -1, -1):
        x[:, k] = dp[:, k] - cp[:, k] * x[:, k + 1]
    return x


def adi_step(omega, u, v, dt, h, nu, bc_old, bc_new, f=None):
    c = dt / 2.0
    a_W, a_P, a_E = coefficients_x(u, h, nu)
    b_S, b_P, b_N = coefficients_y(v, h, nu)
    f_int = 0.0 if f is None else f[1:-1, 1:-1]

    # ===== PASSO 1: implícito em x, explícito em y =====
    omega_half = omega.copy()

    dg = bc_new - bc_old
    for col in (0, -1):
        vb = v[1:-1, col]
        Ly_dg = (
            (nu / h**2 + vb / (2 * h)) * dg[:-2, col]
            - (2 * nu / h**2) * dg[1:-1, col]
            + (nu / h**2 - vb / (2 * h)) * dg[2:, col]
        )
        omega_half[1:-1, col] = 0.5 * (bc_old[1:-1, col] + bc_new[1:-1, col]) - 0.5 * c * Ly_dg

    Ly = b_S * omega[:-2, 1:-1] + b_P * omega[1:-1, 1:-1] + b_N * omega[2:, 1:-1]
    rhs = omega[1:-1, 1:-1] + c * Ly + c * f_int
    rhs[:, 0] += c * a_W[:, 0] * omega_half[1:-1, 0]
    rhs[:, -1] += c * a_E[:, -1] * omega_half[1:-1, -1]

    omega_half[1:-1, 1:-1] = thomas(-c * a_W[:, 1:], 1 - c * a_P, -c * a_E[:, :-1], rhs)

    # ===== PASSO 2: implícito em y, explícito em x =====
    omega_new = omega_half.copy()
    omega_new[0, :] = bc_new[0, :]
    omega_new[-1, :] = bc_new[-1, :]
    omega_new[:, 0] = bc_new[:, 0]
    omega_new[:, -1] = bc_new[:, -1]

    Lx = a_W * omega_half[1:-1, :-2] + a_P * omega_half[1:-1, 1:-1] + a_E * omega_half[1:-1, 2:]
    rhs = omega_half[1:-1, 1:-1] + c * Lx + c * f_int
    rhs[0, :] += c * b_S[0, :] * omega_new[0, 1:-1]
    rhs[-1, :] += c * b_N[-1, :] * omega_new[-1, 1:-1]

    omega_new[1:-1, 1:-1] = thomas(-c * b_S[1:, :].T, 1 - c * b_P.T, -c * b_N[:-1, :].T, rhs.T).T

    return omega_new
