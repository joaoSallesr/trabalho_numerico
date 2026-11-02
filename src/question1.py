import os
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .adi import adi_step
from .exact_solutions import omega_exact, u_exact, v_exact
from .grid import create_grid

Re = 20
T = 1.0
N_LIST = [4, 8, 16, 32, 64]
OUT = os.path.join(os.path.dirname(__file__), "..", "figs")


def solve(N, dt_fun, Re=Re, T=T):
    x, y, X, Y, h = create_grid(N)
    dt0 = dt_fun(h)
    nsteps = int(round(T / dt0))
    dt = T / nsteps  # garante chegar exatamente em T
    nu = 1.0 / Re

    t = 0.0
    omega = omega_exact(X, Y, t, Re)  # condição inicial

    for _ in range(nsteps):
        tm = t + dt / 2
        u = u_exact(X, Y, tm, Re)  # velocidade em t + dt/2
        v = v_exact(X, Y, tm, Re)
        bc_old = omega_exact(X, Y, t, Re)
        bc_new = omega_exact(X, Y, t + dt, Re)
        omega = adi_step(omega, u, v, dt, h, nu, bc_old, bc_new, f=None)  # f = 0
        t += dt

    err = np.max(np.abs(omega - omega_exact(X, Y, T, Re)))
    return X, Y, omega, err


def plot_comparison(Ns=(4, 8, 16, 32, 64)):
    os.makedirs(OUT, exist_ok=True)
    ncols = 3
    nrows = math.ceil((len(Ns) + 1) / ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        subplot_kw={"projection": "3d"},
    )
    axs = np.atleast_1d(axs).ravel()
    xe = np.linspace(-1, 1, 101)
    XE, YE = np.meshgrid(xe, xe)
    axs[0].plot_surface(XE, YE, omega_exact(XE, YE, T, Re), cmap="viridis")
    axs[0].set_title("Exata (T=1)")
    for ax, N in zip(axs[1:], Ns):
        X, Y, w, e = solve(N, lambda h: h)
        ax.plot_surface(X, Y, w, cmap="viridis")
        ax.set_title(f"ADI {N}x{N}  (erro={e:.1e})")
    for ax in axs[len(Ns) + 1 :]:
        ax.set_visible(False)
    for ax in axs:
        if not ax.get_visible():
            continue
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q1a_comparacao.png"), dpi=150)
    plt.close()


def convergence(label, dt_fun):
    errs = []
    for N in N_LIST:
        errs.append(solve(N, dt_fun)[3])
    errs = np.array(errs)
    orders = np.log2(errs[:-1] / errs[1:])
    print(f"\n{label}")
    print(f"{'N':>4} {'h':>10} {'erro_max':>12} {'ordem':>8}")
    for k, N in enumerate(N_LIST):
        o = "" if k == 0 else f"{orders[k - 1]:8.3f}"
        print(f"{N:>4} {2 / N:10.4f} {errs[k]:12.4e} {o:>8}")
    return errs


if __name__ == "__main__":
    plot_comparison()

    cases = {
        "dt = h": lambda h: h,
        "dt = h^2": lambda h: h**2,
    }
    os.makedirs(OUT, exist_ok=True)
    plt.figure(figsize=(8, 5.5))
    h_arr = 2 / np.array(N_LIST)
    for label, fn in cases.items():
        e = convergence(label, fn)
        plt.loglog(h_arr, e, "x-", label=label)
    plt.loglog(h_arr, h_arr**2 * (e[0] / h_arr[0] ** 2), "k--", label="O(h²)")
    plt.xlabel("h")
    plt.ylabel("erro (norma do máximo)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q1b_convergencia.png"), dpi=150)
