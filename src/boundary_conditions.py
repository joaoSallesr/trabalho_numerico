from .exact_solutions import omega_exact


def apply_bc(omega, X, Y, t, Re):

    omega[0, :] = omega_exact(X[0, :], Y[0, :], t, Re)
    omega[-1, :] = omega_exact(X[-1, :], Y[-1, :], t, Re)
    omega[:, 0] = omega_exact(X[:, 0], Y[:, 0], t, Re)
    omega[:, -1] = omega_exact(X[:, -1], Y[:, -1], t, Re)

    return omega
