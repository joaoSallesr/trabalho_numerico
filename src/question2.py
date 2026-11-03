"""Questão 2 — escoamento em cavidade com tampa deslizante (lid-driven cavity flow).

Equações (2)-(3) de TrabNum.pdf em Ω = [0, 1] × [0, 1]:

    ∂ω/∂t − Δω + Re (∂ψ/∂y ∂ω/∂x − ∂ψ/∂x ∂ω/∂y) = 0          (2)
    ∂ψ/∂t − Δψ = ω                                             (3)

O sistema é avançado em pseudo-tempo até o regime quase-estacionário
(∂ω/∂t → 0 e ∂ψ/∂t → 0) com o método ADI (Peaceman–Rachford) e diferenças
centrais. É o mesmo esquema de `adi.adi_step` (Questão 1) com
u = Re ∂ψ/∂y, v = −Re ∂ψ/∂x e ν = 1, escrito com a notação σ, ρx, ρy dos slides
(Aula5-cavity, slides 14-22) e com um solver tridiagonal do LAPACK
(a equivalência com `adi_step` está em `check_adi_equivalence`).

Estratégia (slide 21), a cada passo n → n+1:
  1. ψ^{n+1}: ADI da equação (3) com fonte ω^n e ψ = 0 nos contornos;
  2. condição de contorno de Thom para ω^{n+1} a partir de ψ^{n+1} (slide 20);
  3. ω^{n+1}: ADI da equação (2) com ρx, ρy calculados com ψ^{n+1};
  4. critério de parada max(‖R_ψ‖∞, ‖R_ω‖∞) ≤ tol (resíduos estacionários do slide 22).

Convenção dos arrays: f[j, i] = f(x_i, y_j) (mesma de `grid.create_grid`).
Com u = ∂ψ/∂y e a tampa movendo-se em +x, o vórtice primário gira no sentido
horário, então ψ < 0 e ω < 0 no seu centro (as tabelas da literatura trazem módulos).

Uso (a partir da raiz do projeto):
    python -m src.question2                 # roda os casos que faltam e gera tabelas, figuras e animações
    python -m src.question2 --jobs 3        # idem, rodando até 3 casos em paralelo
    python -m src.question2 --only-post     # só pós-processamento (usa results/*.npz)
    python -m src.question2 --run 1000 128 1.0   # roda um caso (Re, N, Δt/h²) e salva
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.linalg import lapack

from .ghia_data import (
    GHIA_PONTOS_X,
    GHIA_PONTOS_Y,
    GHIA_U,
    GHIA_V,
    PRIMARY_VORTEX_RE1000,
    TABLE_RE1000,
)
from .grid import create_grid

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIGS_DIR = os.path.join(ROOT, "figs")
ANIM_DIR = os.path.join(FIGS_DIR, "animacoes")
RESULTS_DIR = os.path.join(ROOT, "results")
TABLES_DIR = os.path.join(ROOT, "tables")

TOL = 1e-6
CRITERION = "max(||R_psi||_inf, ||R_omega||_inf) <= tol (residuos estacionarios, slide 22)"


class Case(NamedTuple):
    Re: float
    N: int
    sigma: float  # Δt / h²
    max_iter: int = 2_000_000
    tol: float = TOL


# Item a/c: comparação com Ghia et al. para vários Re. Ghia usou 129 x 129 pontos (N = 128)
# até Re = 3200 e 257 x 257 (N = 256) a partir de Re = 5000. Δt = σ h² é limitado pela
# estabilidade do acoplamento ψ-ω: σ ≤ 1 para Re ≤ 1000 em N = 128 e menor para Re maiores
# (Re ≥ 3200 diverge em N = 64 e Re ≥ 5000 em N = 128). Para Re ≥ 5000 a convergência é muito
# lenta (e Re = 10000 está acima da bifurcação de Hopf, Re ≈ 8000), então usa-se tol = 1e-4.
GHIA_CASES = [
    Case(100, 128, 1.0),
    Case(400, 128, 1.0),
    Case(1000, 128, 1.0),
    Case(3200, 128, 0.25),
    Case(5000, 256, 0.5, 1_000_000, 1e-4),
    Case(7500, 256, 0.25, 1_000_000, 1e-4),
    Case(10000, 256, 0.1, 1_000_000, 1e-4),
]

# Item b: Re = 1000 com as malhas e passos de tempo da tabela do slide 23.
RE1000_CASES = [
    Case(1000, 32, 0.1),
    Case(1000, 64, 0.5),
    Case(1000, 128, 1.0),
    Case(1000, 256, 1.0),
]


# ---------------------------------------------------------------------------
# Sistemas tridiagonais
# ---------------------------------------------------------------------------
class ConstantTridiagonal:
    """Fatoração LU (LAPACK ?gttrf) de tridiag(off, diag, off) de ordem n.

    Usada na equação de ψ, cuja matriz é a mesma em todas as linhas e em todos
    os passos: fatora uma vez e resolve muitos lados direitos com ?gttrs.
    """

    def __init__(self, n, diag, off):
        dl, d, du, du2, ipiv, info = lapack.dgttrf(np.full(n - 1, off), np.full(n, diag), np.full(n - 1, off))
        if info != 0:
            raise np.linalg.LinAlgError(f"dgttrf falhou (info={info})")
        self._lu = (dl, d, du, du2, ipiv)

    def solve(self, b):
        """Resolve T x = b para cada coluna de b (sistemas ao longo do eixo 0)."""
        x, info = lapack.dgttrs(*self._lu, b)
        if info != 0:
            raise np.linalg.LinAlgError(f"dgttrs falhou (info={info})")
        return x


def solve_rows(lower, diag, upper, rhs):
    """Resolve um sistema tridiagonal independente por linha (ao longo do eixo 1).

    Na equação k de cada linha: lower[:, k] x[k-1] + diag[:, k] x[k] + upper[:, k] x[k+1] = rhs[:, k]
    (lower[:, 0] e upper[:, -1] são ignorados; a contribuição do contorno deve estar em rhs).
    As linhas são concatenadas em um único sistema tridiagonal desacoplado e resolvidas
    de uma vez pelo LAPACK (?gtsv). Os argumentos são sobrescritos.
    """
    m, n = diag.shape
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0
    _, _, _, x, info = lapack.dgtsv(
        lower.ravel()[1:],
        diag.ravel(),
        upper.ravel()[:-1],
        rhs.reshape(-1, 1),
        overwrite_dl=1,
        overwrite_d=1,
        overwrite_du=1,
        overwrite_b=1,
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"dgtsv falhou (info={info})")
    return x.reshape(m, n)


# ---------------------------------------------------------------------------
# Passos ADI (slides 14-16)
# ---------------------------------------------------------------------------
def adi_psi(psi, omega, sigma, dt, solver):
    """Um passo ADI da equação (3), ∂ψ/∂t − Δψ = ω, com ψ = 0 no contorno (slide 14).

    (1+σ)ψ*_{i,j} − σ/2 (ψ*_{i+1,j} + ψ*_{i-1,j}) = Δt/2 ω^n + (1−σ)ψ^n + σ/2 (ψ^n_{i,j+1} + ψ^n_{i,j-1})
    (1+σ)ψ^{n+1}_{i,j} − σ/2 (ψ^{n+1}_{i,j+1} + ψ^{n+1}_{i,j-1}) = Δt/2 ω^n + (1−σ)ψ* + σ/2 (ψ*_{i+1,j} + ψ*_{i-1,j})
    """
    s2 = 0.5 * sigma
    source = (0.5 * dt) * omega[1:-1, 1:-1]

    # passo 1: implícito em x (cada linha j é um sistema) -> resolve T X^T = rhs^T
    rhs = source + (1.0 - sigma) * psi[1:-1, 1:-1] + s2 * (psi[2:, 1:-1] + psi[:-2, 1:-1])
    half = np.zeros_like(psi)
    half[1:-1, 1:-1] = solver.solve(rhs.T).T

    # passo 2: implícito em y (cada coluna i é um sistema)
    rhs = source + (1.0 - sigma) * half[1:-1, 1:-1] + s2 * (half[1:-1, 2:] + half[1:-1, :-2])
    new = np.zeros_like(psi)
    new[1:-1, 1:-1] = solver.solve(rhs)
    return new


def thom_bc(psi, omega, h, U=1.0):
    """Condição de contorno de Thom para ω (slides 18-20), com ψ = 0 nas paredes.

    Altera apenas as linhas/colunas de contorno de `omega`. Nos cantos superiores
    vale a fórmula da tampa (os cantos não entram no estêncil de nenhum ponto interior).
    """
    omega[:, 0] = -2.0 * psi[:, 1] / h**2  # x = 0
    omega[:, -1] = -2.0 * psi[:, -2] / h**2  # x = 1
    omega[0, :] = -2.0 * psi[1, :] / h**2  # y = 0
    omega[-1, :] = -2.0 * psi[-2, :] / h**2 - 2.0 * U / h  # y = 1 (tampa, u = U)
    return omega


def adi_omega(omega, psi, sigma, Re, h, U=1.0):
    """Um passo ADI da equação (2) (slide 16), com ψ = ψ^{n+1} e contorno de Thom.

    ρx = Re (ψ_{i+1,j} − ψ_{i-1,j}) / 4 ,  ρy = Re (ψ_{i,j+1} − ψ_{i,j-1}) / 4

    (1+σ)ω* + σ/2 (ρy−1) ω*_{i+1,j} − σ/2 (ρy+1) ω*_{i-1,j}
        = (1−σ)ω^n + σ/2 (ρx+1) ω^n_{i,j+1} + σ/2 (1−ρx) ω^n_{i,j-1}
    (1+σ)ω^{n+1} − σ/2 (ρx+1) ω^{n+1}_{i,j+1} + σ/2 (ρx−1) ω^{n+1}_{i,j-1}
        = (1−σ)ω* + σ/2 (1−ρy) ω*_{i+1,j} + σ/2 (1+ρy) ω*_{i-1,j}

    Nas paredes x = 0 e x = 1, o valor intermediário ω* usa a mesma correção de
    Peaceman–Rachford de `adi.adi_step`:  ω* = (g^n + g^{n+1})/2 − σ/4 δ_yy(g^{n+1} − g^n).
    """
    s2 = 0.5 * sigma
    rho_x = (0.25 * Re) * (psi[1:-1, 2:] - psi[1:-1, :-2])
    rho_y = (0.25 * Re) * (psi[2:, 1:-1] - psi[:-2, 1:-1])

    new = omega.copy()
    thom_bc(psi, new, h, U)  # valores de contorno em n+1

    half = np.empty_like(omega)
    for col in (0, -1):
        g0 = omega[:, col]
        g1 = new[:, col]
        dg = g1 - g0
        half[1:-1, col] = 0.5 * (g0[1:-1] + g1[1:-1]) - (0.5 * s2) * (dg[2:] - 2.0 * dg[1:-1] + dg[:-2])

    # passo 1: implícito em x, explícito em y
    lower = -s2 * (rho_y + 1.0)  # coeficiente de ω*_{i-1,j}
    upper = s2 * (rho_y - 1.0)  # coeficiente de ω*_{i+1,j}
    rhs = (1.0 - sigma) * omega[1:-1, 1:-1] + s2 * ((1.0 + rho_x) * omega[2:, 1:-1] + (1.0 - rho_x) * omega[:-2, 1:-1])
    rhs[:, 0] -= lower[:, 0] * half[1:-1, 0]
    rhs[:, -1] -= upper[:, -1] * half[1:-1, -1]
    diag = np.full_like(rhs, 1.0 + sigma)
    half[1:-1, 1:-1] = solve_rows(lower, diag, upper, rhs)

    # passo 2: implícito em y, explícito em x (sistemas ao longo das colunas -> transpostos)
    lower = np.ascontiguousarray((s2 * (rho_x - 1.0)).T)  # coeficiente de ω_{i,j-1}
    upper = np.ascontiguousarray((-s2 * (rho_x + 1.0)).T)  # coeficiente de ω_{i,j+1}
    rhs = (1.0 - sigma) * half[1:-1, 1:-1] + s2 * ((1.0 - rho_y) * half[1:-1, 2:] + (1.0 + rho_y) * half[1:-1, :-2])
    rhs = np.ascontiguousarray(rhs.T)
    rhs[:, 0] -= lower[:, 0] * new[0, 1:-1]
    rhs[:, -1] -= upper[:, -1] * new[-1, 1:-1]
    diag = np.full_like(rhs, 1.0 + sigma)
    new[1:-1, 1:-1] = solve_rows(lower, diag, upper, rhs).T
    return new


# ---------------------------------------------------------------------------
# Resíduos e velocidades
# ---------------------------------------------------------------------------
def _laplacian(f, h):
    return (f[1:-1, 2:] + f[1:-1, :-2] + f[2:, 1:-1] + f[:-2, 1:-1] - 4.0 * f[1:-1, 1:-1]) / h**2


def residuals(psi, omega, h, Re):
    """Normas do máximo (pontos interiores) dos resíduos estacionários do slide 22.

    R_ψ = Δ_h ψ + ω
    R_ω = (1/Re) Δ_h ω − (∂ψ/∂y)(∂ω/∂x) + (∂ψ/∂x)(∂ω/∂y)   (diferenças centrais)
    """
    r_psi = _laplacian(psi, h) + omega[1:-1, 1:-1]
    psi_x = psi[1:-1, 2:] - psi[1:-1, :-2]
    psi_y = psi[2:, 1:-1] - psi[:-2, 1:-1]
    w_x = omega[1:-1, 2:] - omega[1:-1, :-2]
    w_y = omega[2:, 1:-1] - omega[:-2, 1:-1]
    r_omega = _laplacian(omega, h) / Re + (psi_x * w_y - psi_y * w_x) / (4.0 * h * h)
    return float(np.max(np.abs(r_psi))), float(np.max(np.abs(r_omega)))


def velocities(psi, h, U=1.0):
    """u = ∂ψ/∂y e v = −∂ψ/∂x por diferenças centrais (2ª ordem) nos pontos interiores.

    Nos contornos valem as condições de não deslizamento: u = U na tampa (y = 1),
    u = v = 0 nas demais paredes.
    """
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * h)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * h)
    u[-1, 1:-1] = U
    return u, v


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class CavityResult:
    Re: float
    N: int
    h: float
    dt: float
    sigma: float
    tol: float
    criterion: str
    iterations: int
    status: str  # "convergiu", "max_iter" ou "divergiu"
    res_psi: float
    res_omega: float
    t_final: float
    cpu_time: float
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    psi: np.ndarray = field(repr=False)
    omega: np.ndarray = field(repr=False)
    history: np.ndarray = field(repr=False)  # colunas: iteração, ‖R_ψ‖∞, ‖R_ω‖∞
    frames: dict = field(default=None, repr=False)

    @property
    def converged(self):
        return self.status == "convergiu"

    def metadata(self):
        return {
            "Re": self.Re,
            "N": self.N,
            "malha": f"{self.N}x{self.N}",
            "h": self.h,
            "dt": self.dt,
            "dt/h^2": self.sigma,
            "tolerancia": self.tol,
            "criterio_parada": self.criterion,
            "iteracoes": self.iterations,
            "status": self.status,
            "res_psi": self.res_psi,
            "res_omega": self.res_omega,
            "t_final": self.t_final,
            "tempo_cpu_s": self.cpu_time,
        }


def solve_cavity(
    Re,
    N,
    sigma,
    tol=TOL,
    max_iter=2_000_000,
    U=1.0,
    check_every=25,
    divergence_limit=1e8,
    frames=False,
    max_frames=200,
    log_every=None,
    log=print,
    initial=None,
):
    """Avança (2)-(3) em pseudo-tempo, a partir de ψ = ω = 0, até o critério de parada.

    Com `initial` (um CavityResult salvo, de mesmo Re, N e σ), continua a evolução a partir
    do estado final dele, mantendo a contagem de iterações, o histórico e os quadros.

    sigma = Δt/h² (os slides expressam Δt como fração de h²). O resíduo é verificado a
    cada `check_every` passos e em todos os passos quando já está abaixo de 2·tol
    (o decaimento de 2·tol até tol leva muito mais que `check_every` passos), de modo
    que o número de iterações informado é exato.
    Com `frames=True`, guarda instantâneos de ψ e ω (espaçados geometricamente em
    iterações, no máximo ~129 pontos por direção) para as animações.
    """
    x, y, _, _, h = create_grid(N, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    dt = sigma * h * h
    psi = np.zeros((N + 1, N + 1))
    omega = np.zeros((N + 1, N + 1))
    psi_solver = ConstantTridiagonal(N - 1, 1.0 + sigma, -0.5 * sigma)

    hist = []
    snaps = {"iteracao": [], "psi": [], "omega": [], "residuo": []}
    stride = max(1, N // 128)  # subamostragem espacial dos quadros
    # primeiro quadro em t' = Re t ≈ 0.25 (tempo físico), depois razão geométrica
    next_frame = max(1, int(round(0.25 / (Re * dt))))
    ratio = 1.05

    def add_frame(it, res_max):
        snaps["iteracao"].append(it)
        snaps["psi"].append(psi[::stride, ::stride].astype(np.float32))
        snaps["omega"].append(omega[::stride, ::stride].astype(np.float32))
        snaps["residuo"].append(res_max)

    status = "max_iter"
    res = (np.inf, np.inf)
    last_res = np.inf
    it = start = 0
    cpu_before = 0.0
    if initial is not None:
        if (initial.N, initial.sigma) != (N, sigma) or not np.isclose(initial.Re, Re):
            raise ValueError("o resultado inicial não corresponde a (Re, N, sigma)")
        psi, omega = initial.psi.copy(), initial.omega.copy()
        it = start = initial.iterations
        cpu_before = initial.cpu_time
        hist = [tuple(r) for r in initial.history]
        if len(hist):
            res = (hist[-1][1], hist[-1][2])
            last_res = max(res)
        if frames and initial.frames is not None and len(initial.frames["iteracao"]):
            snaps = {k: list(v) for k, v in initial.frames.items()}
            its = snaps["iteracao"]
            if len(its) >= 3:
                ratio = max(1.05, its[-2] / its[-3])
            next_frame = int(np.ceil(its[-1] * ratio))
    t0 = time.perf_counter()
    for it in range(start + 1, max_iter + 1):
        psi = adi_psi(psi, omega, sigma, dt, psi_solver)
        omega = adi_omega(omega, psi, sigma, Re, h, U)

        if it % check_every == 0 or last_res < 2.0 * tol or it == max_iter:
            res = residuals(psi, omega, h, Re)
            last_res = max(res)
            hist.append((it, res[0], res[1]))
            if not np.isfinite(last_res) or last_res > divergence_limit:
                status = "divergiu"
                break
            if last_res <= tol:
                status = "convergiu"
                break

        if frames and it >= next_frame:
            add_frame(it, last_res)
            next_frame = max(it + 1, int(np.ceil(it * ratio)))
            if len(snaps["iteracao"]) > max_frames:  # descarta um a cada dois
                for k in snaps:
                    snaps[k] = snaps[k][::2]
                ratio = ratio**2

        if log_every and it % log_every == 0:
            log(
                f"  Re={Re:g} N={N} it={it} t={it * dt:.4f} "
                f"R_psi={res[0]:.2e} R_omega={res[1]:.2e} ({time.perf_counter() - t0:.0f} s)"
            )

    cpu = cpu_before + time.perf_counter() - t0
    if frames and status != "divergiu" and (not snaps["iteracao"] or snaps["iteracao"][-1] != it):
        add_frame(it, max(res))

    return CavityResult(
        Re=float(Re),
        N=int(N),
        h=float(h),
        dt=float(dt),
        sigma=float(sigma),
        tol=float(tol),
        criterion=CRITERION,
        iterations=int(it),
        status=status,
        res_psi=float(res[0]),
        res_omega=float(res[1]),
        t_final=float(it * dt),
        cpu_time=float(cpu),
        x=x,
        y=y,
        psi=psi,
        omega=omega,
        history=np.array(hist, dtype=float).reshape(-1, 3),
        frames={k: np.array(v) for k, v in snaps.items()} if frames else None,
    )


def check_adi_equivalence(Re=1000.0, N=32, sigma=0.1, steps=200):
    """Compara `adi_psi`/`adi_omega` com `adi.adi_step` (Questão 1) passo a passo.

    Retorna as maiores diferenças em ψ e ω (devem estar no nível do arredondamento).
    """
    from .adi import adi_step

    _, _, _, _, h = create_grid(N, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    dt = sigma * h * h
    psi_a = np.zeros((N + 1, N + 1))
    w_a = np.zeros_like(psi_a)
    psi_b, w_b = psi_a.copy(), w_a.copy()
    solver = ConstantTridiagonal(N - 1, 1.0 + sigma, -0.5 * sigma)
    for _ in range(steps):
        psi_a = adi_step(psi_a, None, None, dt, h, 1.0, 0.0, 0.0, f=w_a)
        u, v = velocities(psi_a, h)
        w_a = adi_step(w_a, Re * u, Re * v, dt, h, 1.0, w_a, thom_bc(psi_a, w_a.copy(), h))
        psi_b = adi_psi(psi_b, w_b, sigma, dt, solver)
        w_b = adi_omega(w_b, psi_b, sigma, Re, h)
    return float(np.max(np.abs(psi_a - psi_b))), float(np.max(np.abs(w_a - w_b)))


# ---------------------------------------------------------------------------
# Persistência dos resultados (results/*.npz)
# ---------------------------------------------------------------------------
def case_name(Re, N, sigma, tol=TOL):
    name = f"cavidade_Re{Re:g}_N{N}_s{sigma:g}"
    return name if np.isclose(tol, TOL) else name + f"_tol{tol:.0e}"


def result_path(case, folder=RESULTS_DIR):
    return os.path.join(folder, case_name(case.Re, case.N, case.sigma, case.tol) + ".npz")


def save_result(res, folder=RESULTS_DIR):
    """Salva campos, histórico e metadados (.npz + .json); os quadros da animação vão num arquivo à parte."""
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, case_name(res.Re, res.N, res.sigma, res.tol))
    np.savez_compressed(
        base + ".npz",
        x=res.x,
        y=res.y,
        psi=res.psi,
        omega=res.omega,
        historico=res.history,
        meta=json.dumps(res.metadata()),
    )
    with open(base + ".json", "w", encoding="utf-8") as fp:
        json.dump(res.metadata(), fp, indent=2, ensure_ascii=False)
    if res.frames is not None:
        np.savez_compressed(base + "_quadros.npz", **res.frames)
    return base + ".npz"


def load_result(path, with_frames=False):
    with np.load(path) as d:
        meta = json.loads(str(d["meta"]))
        arrays = {k: d[k] for k in ("x", "y", "psi", "omega")}
        history = d["historico"]
    frames = None
    frames_path = path.replace(".npz", "_quadros.npz")
    if with_frames and os.path.exists(frames_path):
        with np.load(frames_path) as q:
            frames = {k: q[k] for k in q.files}
    return CavityResult(
        Re=meta["Re"],
        N=meta["N"],
        h=meta["h"],
        dt=meta["dt"],
        sigma=meta["dt/h^2"],
        tol=meta["tolerancia"],
        criterion=meta["criterio_parada"],
        iterations=meta["iteracoes"],
        status=meta["status"],
        res_psi=meta["res_psi"],
        res_omega=meta["res_omega"],
        t_final=meta["t_final"],
        cpu_time=meta["tempo_cpu_s"],
        history=history,
        frames=frames,
        **arrays,
    )


def _run_and_save(case, frames=True, folder=RESULTS_DIR, log_every=20000, resume=False):
    def log(msg):
        print(msg, flush=True)

    initial = None
    path = result_path(case, folder)
    if resume and os.path.exists(path):
        initial = load_result(path, with_frames=frames)
        if initial.status == "convergiu":
            log(f"[já convergido] {os.path.basename(path)}")
            return path
        if initial.status == "divergiu":
            initial = None
    if initial is not None:
        log(f"[retomando] {os.path.basename(path)} a partir da iteração {initial.iterations}")
    else:
        log(f"[início] Re={case.Re:g} malha {case.N}x{case.N} Δt={case.sigma:g} h² tol={case.tol:g}")
    res = solve_cavity(
        case.Re,
        case.N,
        case.sigma,
        tol=case.tol,
        max_iter=case.max_iter,
        frames=frames,
        log_every=log_every,
        log=log,
        initial=initial,
    )
    path = save_result(res, folder)
    log(
        f"[fim] Re={case.Re:g} malha {case.N}x{case.N}: {res.status} em {res.iterations} iterações "
        f"(R_psi={res.res_psi:.2e}, R_omega={res.res_omega:.2e}, {res.cpu_time / 60:.1f} min)"
    )
    return path


def run_cases(cases, jobs=1, force=False, frames=True, folder=RESULTS_DIR):
    """Roda os casos que ainda não têm resultado salvo (ou todos, com force=True)."""
    todo = []
    for c in dict.fromkeys(cases):  # remove duplicados mantendo a ordem
        if force or not os.path.exists(result_path(c, folder)):
            todo.append(c)
    if not todo:
        return
    print(f"Rodando {len(todo)} caso(s) com {min(jobs, len(todo))} processo(s)...", flush=True)
    if jobs > 1 and len(todo) > 1:
        # casos mais caros primeiro, para equilibrar os processos
        todo.sort(key=lambda c: c.N**4 / c.sigma, reverse=True)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_run_and_save, c, frames, folder) for c in todo]
            for f in as_completed(futures):
                f.result()
    else:
        for c in todo:
            _run_and_save(c, frames, folder)


def load_cases(cases, folder=RESULTS_DIR, with_frames=False):
    out = {}
    for c in cases:
        p = result_path(c, folder)
        if os.path.exists(p):
            out[c] = load_result(p, with_frames=with_frames)
        else:
            print(f"aviso: resultado ausente para {c} ({p})")
    return out


# ---------------------------------------------------------------------------
# Pós-processamento
# ---------------------------------------------------------------------------
def center_line(field_, coord, axis):
    """Valores de `field_` na linha coord = 0.5 (axis=1: x = 0.5; axis=0: y = 0.5)."""
    h = coord[1] - coord[0]
    k = 0.5 / h
    i0 = int(np.floor(k + 1e-9))
    w = k - i0
    take = (lambda i: field_[:, i]) if axis == 1 else (lambda i: field_[i, :])
    return take(i0) if w < 1e-9 else (1.0 - w) * take(i0) + w * take(i0 + 1)


def ghia_profiles(res):
    """u em x = 0.5 (pontos y de Ghia) e v em y = 0.5 (pontos x de Ghia).

    As coordenadas exatas dos pontos de Ghia são (ponto - 1)/128, que coincidem com nós
    das malhas N = 128 e 256; para outras malhas usa-se interpolação linear.
    """
    u, v = velocities(res.psi, res.h)
    y_pts = (GHIA_PONTOS_Y - 1) / 128.0
    x_pts = (GHIA_PONTOS_X - 1) / 128.0
    u_adi = np.interp(y_pts, res.y, center_line(u, res.x, axis=1))
    v_adi = np.interp(x_pts, res.x, center_line(v, res.y, axis=0))
    key = int(round(res.Re))
    return {
        "y": y_pts,
        "u": u_adi,
        "u_ghia": GHIA_U.get(key),
        "x": x_pts,
        "v": v_adi,
        "v_ghia": GHIA_V.get(key),
    }


def primary_vortex(res):
    """Maior |ψ| nos nós da malha e ω no mesmo ponto (como na tabela do slide 23)."""
    j, i = np.unravel_index(np.argmax(np.abs(res.psi)), res.psi.shape)
    return {"psi": float(res.psi[j, i]), "omega": float(res.omega[j, i]), "x": float(res.x[i]), "y": float(res.y[j])}


def dt_label(sigma, latex=False):
    """Δt como fração de h² (ex.: 'h²/2', '0.25 h²')."""
    h2 = "h^2" if latex else "h²"
    if np.isclose(sigma, 1.0):
        return h2
    inv = 1.0 / sigma
    if np.isclose(inv, round(inv)):
        return f"{h2}/{round(inv)}"
    return f"{sigma:g} {h2}" if not latex else f"{sigma:g}\\,{h2}"


def _tex_num(v, fmt):
    if v is None or not np.isfinite(v):
        return "---"
    s = format(v, fmt)
    return f"${s}$" if s.startswith("-") else s


def _sci_tex(v, digits=1):
    m, e = f"{v:.{digits}e}".split("e")
    if digits == 0 and m == "1":
        return f"$10^{{{int(e)}}}$"
    return f"${m}\\times 10^{{{int(e)}}}$"


def _coord(v):
    """Coordenada com 4 casas, arredondando metade para cima (0.53125 -> 0.5313, como Ghia e o slide 23)."""
    return f"{v + 1e-9:.4f}"


STATUS_PT = {"convergiu": "convergiu", "max_iter": "não convergiu (máx. iterações)", "divergiu": "divergiu"}
STATUS_TEX = {"convergiu": "convergiu", "max_iter": "não convergiu", "divergiu": "divergiu"}

# Valores de Ghia et al. (1982) provavelmente com erro de impressão: (componente, Re, ponto da malha 129).
# Re = 400, v em x = 0.9063: tabelado -0.23827, mas os vizinhos (que concordam com o ADI até ~0.004)
# indicam ~ -0.38. Esses pontos são marcados nas tabelas/figuras e excluídos do erro máximo.
GHIA_SUSPECT = {("v", 400, 117)}
SUSPECT_NOTE = "valor de Ghia et al. provavelmente com erro de impressão (os pontos vizinhos concordam com o ADI)"


def _is_suspect(comp, Re, pt):
    return (comp, int(round(Re)), int(pt)) in GHIA_SUSPECT


def _max_err(prof_r, comp, pts, Re):
    """Maior |ADI − Ghia| ignorando os valores suspeitos; retorna (erro, excluiu_algum)."""
    g = prof_r[f"{comp}_ghia"]
    if g is None:
        return None, False
    keep = np.array([not _is_suspect(comp, Re, p) for p in pts])
    return float(np.max(np.abs(prof_r[comp] - g)[keep])), not keep.all()


def write_profile_tables(results, folder=TABLES_DIR):
    """Tabelas da componente u em x = 0.5 e v em y = 0.5 (ADI × Ghia et al.) em .tex e .csv."""
    os.makedirs(folder, exist_ok=True)
    ordered = sorted(results.values(), key=lambda r: r.Re)
    prof = {r.Re: ghia_profiles(r) for r in ordered}
    out = {}
    for comp, coord_name, pts, line in (
        ("u", "y", GHIA_PONTOS_Y, "vertical $x = 0.5$"),
        ("v", "x", GHIA_PONTOS_X, "horizontal $y = 0.5$"),
    ):
        coords = (pts - 1) / 128.0
        # CSV
        csv_path = os.path.join(folder, f"q2a_{comp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            wr = csv.writer(fp)
            header = ["ponto_ghia", coord_name]
            for r in ordered:
                header += [f"{comp}_ADI_Re{r.Re:g}", f"{comp}_Ghia_Re{r.Re:g}"]
            wr.writerow(header)
            for k, p in enumerate(pts):
                row = [int(p), f"{coords[k]:.6f}"]
                for r in ordered:
                    g = prof[r.Re][f"{comp}_ghia"]
                    row += [f"{prof[r.Re][comp][k]:.5f}", "" if g is None else f"{g[k]:.5f}"]
                wr.writerow(row)
        # LaTeX
        ncol = len(ordered)
        lines = [
            "% Gerado por src/question2.py — requer \\usepackage{booktabs} e \\usepackage{graphicx}",
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Componente ${comp}$ da velocidade ao longo da linha {line}: "
            f"método ADI (este trabalho) e Ghia et al. (1982). "
            f"Malhas e passos de tempo de cada caso na Tabela~\\ref{{tab:q2_parametros}}.}}",
            f"\\label{{tab:q2a_{comp}}}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{rr" + "rr" * ncol + "}",
            "\\toprule",
            " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$Re = {r.Re:g}$}}" for r in ordered) + " \\\\",
            "".join(f"\\cmidrule(lr){{{3 + 2 * k}-{4 + 2 * k}}}" for k in range(ncol)),
            f"Ponto & ${coord_name}$ & " + " & ".join("ADI & Ghia" for _ in ordered) + " \\\\",
            "\\midrule",
        ]
        any_suspect = False
        for k, p in enumerate(pts):
            cells = [str(int(p)), _coord(coords[k])]
            for r in ordered:
                g = prof[r.Re][f"{comp}_ghia"]
                ghia_cell = "---" if g is None else _tex_num(g[k], ".5f")
                if g is not None and _is_suspect(comp, r.Re, p):
                    ghia_cell += "$^{*}$"
                    any_suspect = True
                cells += [_tex_num(prof[r.Re][comp][k], ".5f"), ghia_cell]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
        errs = [_max_err(prof[r.Re], comp, pts, r.Re) for r in ordered]
        lines.append(
            "\\multicolumn{2}{l}{máx.\\ $|\\mathrm{ADI}-\\mathrm{Ghia}|$} & "
            + " & ".join(
                f"\\multicolumn{{2}}{{c}}{{{_tex_num(e, '.5f')}{'$^{*}$' if excl else ''}}}" for e, excl in errs
            )
            + " \\\\"
        )
        lines += ["\\bottomrule", "\\end{tabular}}"]
        if any_suspect:
            lines.append(f"\\par\\smallskip{{\\footnotesize $^{{*}}$ {SUSPECT_NOTE}; excluído do erro máximo.}}")
        lines += ["\\end{table}", ""]
        with open(os.path.join(folder, f"q2a_{comp}.tex"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(lines))
        out[comp] = (coords, prof, [e for e, _ in errs])
    return out


def print_profile_tables(results):
    ordered = sorted(results.values(), key=lambda r: r.Re)
    prof = {r.Re: ghia_profiles(r) for r in ordered}
    for comp, coord_name, pts in (("u", "y", GHIA_PONTOS_Y), ("v", "x", GHIA_PONTOS_X)):
        where = "x = 0.5" if comp == "u" else "y = 0.5"
        print(f"\nComponente {comp} em {where} (ADI | Ghia et al.)")
        print(f"{'pt':>4} {coord_name:>7} " + " ".join(f"{'Re=' + format(r.Re, 'g'):>19}" for r in ordered))
        for k, p in enumerate(pts):
            cells = []
            for r in ordered:
                g = prof[r.Re][f"{comp}_ghia"]
                mark = "*" if g is not None and _is_suspect(comp, r.Re, p) else " "
                cells.append(f"{prof[r.Re][comp][k]:9.5f}|{'' if g is None else format(g[k], '9.5f'):>9}{mark}")
            print(f"{int(p):>4} {_coord((p - 1) / 128):>7} " + " ".join(cells))
        errs = []
        for r in ordered:
            e, excl = _max_err(prof[r.Re], comp, pts, r.Re)
            errs.append("" if e is None else f"{e:.5f}{'*' if excl else ' '}")
        print(f"{'máx |ADI-Ghia|':>12} " + " ".join(f"{e:>20}" for e in errs))
        if any(_is_suspect(comp, r.Re, p) for r in ordered for p in pts):
            print(f"  * {SUSPECT_NOTE}; excluído do erro máximo")


def write_parameters_table(results, folder=TABLES_DIR, name="q2_parametros"):
    """Dados de cada simulação: Re, malha, h, Δt, iterações, critério, tolerância, resíduos..."""
    os.makedirs(folder, exist_ok=True)
    rows = sorted(results.values(), key=lambda r: (r.Re, r.N))
    with open(os.path.join(folder, name + ".csv"), "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(
            [
                "Re",
                "malha",
                "h",
                "dt",
                "dt/h^2",
                "iteracoes",
                "t_final",
                "criterio_parada",
                "tolerancia",
                "res_psi",
                "res_omega",
                "status",
                "tempo_cpu_s",
            ]
        )
        for r in rows:
            wr.writerow(
                [
                    f"{r.Re:g}",
                    f"{r.N}x{r.N}",
                    f"{r.h:.8g}",
                    f"{r.dt:.6e}",
                    f"{r.sigma:g}",
                    r.iterations,
                    f"{r.t_final:.6f}",
                    r.criterion,
                    f"{r.tol:g}",
                    f"{r.res_psi:.3e}",
                    f"{r.res_omega:.3e}",
                    r.status,
                    f"{r.cpu_time:.1f}",
                ]
            )
    lines = [
        "% Gerado por src/question2.py — requer \\usepackage{booktabs} e \\usepackage{graphicx}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Dados das simulações da cavidade. Critério de parada: "
        "$\\max(\\|R_\\psi^{n+1}\\|_\\infty, \\|R_\\omega^{n+1}\\|_\\infty) \\le \\mathrm{tol}$, "
        "com os resíduos estacionários do slide 22 calculados nos pontos interiores; "
        "condição inicial $\\psi = \\omega = 0$; $t_{\\mathrm{final}} = n\\,\\Delta t$ é o pseudo-tempo "
        "ao final das $n$ iterações.}",
        f"\\label{{tab:{name}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{rclcrrrcccl}",
        "\\toprule",
        "$Re$ & Malha & $h$ & \\multicolumn{2}{c}{$\\Delta t$} & Iterações & $t_{\\mathrm{final}}$ & tol & "
        "$\\|R_\\psi\\|_\\infty$ & $\\|R_\\omega\\|_\\infty$ & Situação \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r.Re:g} & ${r.N}\\times{r.N}$ & $1/{r.N}$ & ${dt_label(r.sigma, latex=True)}$ & "
            f"{_sci_tex(r.dt, 3)} & {r.iterations} & {r.t_final:.4f} & {_sci_tex(r.tol, 0)} & "
            f"{_sci_tex(r.res_psi)} & {_sci_tex(r.res_omega)} & {STATUS_TEX[r.status]} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    with open(os.path.join(folder, name + ".tex"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))


def print_parameters_table(results, title):
    print(f"\n{title}")
    print(
        f"{'Re':>6} {'malha':>9} {'h':>9} {'dt':>11} {'dt/h²':>6} {'iterações':>10} {'t_final':>8} "
        f"{'tol':>6} {'R_psi':>9} {'R_omega':>9} {'CPU (min)':>9}  situação"
    )
    for r in sorted(results.values(), key=lambda r: (r.Re, r.N)):
        print(
            f"{r.Re:6g} {r.N:4d}x{r.N:<4d} {r.h:9.6f} {r.dt:11.4e} {r.sigma:6g} {r.iterations:10d} {r.t_final:8.4f} "
            f"{r.tol:6.0e} {r.res_psi:9.2e} {r.res_omega:9.2e} {r.cpu_time / 60:9.1f}  {r.status}"
        )
    print(f"critério de parada: {CRITERION}")


def write_re1000_table(results, folder=TABLES_DIR):
    """Item b: vórtice primário para Re = 1000 — ADI × tabela do slide 23 × literatura."""
    os.makedirs(folder, exist_ok=True)
    rows = sorted((r for r in results.values() if np.isclose(r.Re, 1000.0)), key=lambda r: r.N)
    data = []
    for r in rows:
        pv = primary_vortex(r)
        ref = TABLE_RE1000.get(f"{r.N}x{r.N}")
        data.append((r, pv, ref))
    with open(os.path.join(folder, "q2b_re1000.csv"), "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(
            [
                "malha",
                "dt",
                "iteracoes",
                "psi_ADI",
                "omega_ADI",
                "x_ADI",
                "y_ADI",
                "psi_slide23",
                "omega_slide23",
                "x_slide23",
                "y_slide23",
            ]
        )
        for r, pv, ref in data:
            wr.writerow(
                [
                    f"{r.N}x{r.N}",
                    dt_label(r.sigma),
                    r.iterations,
                    f"{abs(pv['psi']):.6f}",
                    f"{abs(pv['omega']):.6f}",
                    _coord(pv["x"]),
                    _coord(pv["y"]),
                ]
                + ([f"{ref['psi']:.6f}", f"{ref['omega']:.6f}", f"{ref['x']:.4f}", f"{ref['y']:.4f}"] if ref else [""] * 4)
            )
        wr.writerow([])
        wr.writerow(["referencia", "psi", "omega", "x", "y"])
        for name, ref in PRIMARY_VORTEX_RE1000.items():
            wr.writerow(
                [name, ref["psi"], "" if ref["omega"] is None else ref["omega"], ref["x"], ref["y"]]
            )

    def cell(v, fmt):
        return "---" if v is None else format(v, fmt)

    lines = [
        "% Gerado por src/question2.py — requer \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Vórtice primário para $Re = 1000$: maior valor absoluto de $\\psi$ nos nós da malha, "
        "vorticidade no mesmo ponto e sua localização. Valores em módulo (com $u = \\partial\\psi/\\partial y$ "
        "e a tampa movendo-se em $+x$, $\\psi$ e $\\omega$ são negativos no centro do vórtice).}",
        "\\label{tab:q2b_re1000}",
        "\\footnotesize",
        "\\begin{tabular}{lcrcccccc}",
        "\\toprule",
        "Malha & $\\Delta t$ & Iterações & $|\\psi|_{\\max}$ & $|\\psi|_{\\max}$ (slide 23) & $|\\omega|$ & "
        "$|\\omega|$ (slide 23) & $x$ & $y$ \\\\",
        "\\midrule",
    ]
    for r, pv, ref in data:
        lines.append(
            f"${r.N}\\times{r.N}$ & ${dt_label(r.sigma, latex=True)}$ & {r.iterations} & "
            f"{abs(pv['psi']):.6f} & {cell(ref and ref['psi'], '.6f')} & {abs(pv['omega']):.6f} & "
            f"{cell(ref and ref['omega'], '.6f')} & {_coord(pv['x'])} & {_coord(pv['y'])} \\\\"
        )
    lines += [
        "\\midrule",
        "\\multicolumn{9}{l}{\\emph{Referências (tabela do slide 24)}} \\\\",
    ]
    for name, ref in PRIMARY_VORTEX_RE1000.items():
        tex_name = name.replace("&", r"\&")
        lines.append(
            f"\\multicolumn{{3}}{{l}}{{{tex_name}}} & {ref['psi']} & & "
            f"{'---' if ref['omega'] is None else ref['omega']} & & {ref['x']} & {ref['y']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(os.path.join(folder, "q2b_re1000.tex"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    return data


def print_re1000_table(results):
    data = []
    print("\nItem b — vórtice primário, Re = 1000 (valores em módulo)")
    print(
        f"{'malha':>9} {'dt':>7} {'iter.':>7} {'|psi| ADI':>10} {'slide 23':>9} {'|omega| ADI':>11} "
        f"{'slide 23':>9} {'x':>7} {'y':>7}  (x, y) slide"
    )
    for r in sorted((r for r in results.values() if np.isclose(r.Re, 1000.0)), key=lambda r: r.N):
        pv = primary_vortex(r)
        ref = TABLE_RE1000.get(f"{r.N}x{r.N}")
        data.append((r, pv, ref))
        print(
            f"{r.N:4d}x{r.N:<4d} {dt_label(r.sigma):>7} {r.iterations:7d} {abs(pv['psi']):10.6f} "
            f"{ref['psi'] if ref else float('nan'):9.6f} {abs(pv['omega']):11.6f} "
            f"{ref['omega'] if ref else float('nan'):9.6f} {_coord(pv['x']):>7} {_coord(pv['y']):>7}  "
            + (f"({ref['x']:.4f}, {ref['y']:.4f})" if ref else "")
        )
    print("Referências:")
    for name, ref in PRIMARY_VORTEX_RE1000.items():
        om = "---" if ref["omega"] is None else f"{ref['omega']:.6f}"
        print(f"  {name:32s} psi={ref['psi']:.7f} omega={om:>9} x={ref['x']:.4f} y={ref['y']:.4f}")
    return data


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_C, AXIS_C = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEG_C, POS_C, ZERO_C = "#2a78d6", "#e34948", "#898781"  # polos divergentes (azul < 0 < vermelho)

# Níveis das curvas de nível de Ghia et al. (1982), Tabela III
PSI_LEVELS = np.array(
    [-0.1175, -0.115, -0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01, -1e-4, -1e-5, -1e-7, -1e-10]
    + [1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 1.5e-3, 3e-3]
)
OMEGA_LEVELS = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.edgecolor": AXIS_C,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "xtick.color": AXIS_C,
            "ytick.color": AXIS_C,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "grid.color": GRID_C,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )
    return plt


def _case_label(r):
    tol = "" if np.isclose(r.tol, TOL) else f", tol = {r.tol:.0e}".replace("e-0", "e-")
    return f"Re = {r.Re:g} ({r.N}×{r.N}, Δt = {dt_label(r.sigma)}{tol})"


def plot_profiles(results, folder=FIGS_DIR):
    """Item a: perfis de u em x = 0.5 e de v em y = 0.5 (ADI × Ghia et al.), um painel por Re."""
    plt = _mpl()
    os.makedirs(folder, exist_ok=True)
    ordered = sorted(results.values(), key=lambda r: r.Re)
    paths = []
    for comp in ("u", "v"):
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.6), sharex=True, sharey=True)
        axs = axs.ravel()
        for ax, r in zip(axs, ordered):
            u, v = velocities(r.psi, r.h)
            prof = ghia_profiles(r)
            if comp == "u":
                line = center_line(u, r.x, axis=1)
                ax.plot(line, r.y, color=SERIES[0], lw=1.5, label="ADI")
                if prof["u_ghia"] is not None:
                    ax.plot(
                        prof["u_ghia"], prof["y"], "o", ms=5, color=SERIES[1], mec="white", mew=0.8,
                        label="Ghia et al. (1982)",
                    )
            else:
                line = center_line(v, r.y, axis=0)
                ax.plot(r.x, line, color=SERIES[0], lw=1.5, label="ADI")
                if prof["v_ghia"] is not None:
                    ax.plot(
                        prof["x"], prof["v_ghia"], "o", ms=5, color=SERIES[1], mec="white", mew=0.8,
                        label="Ghia et al. (1982)",
                    )
            pts = GHIA_PONTOS_Y if comp == "u" else GHIA_PONTOS_X
            g = prof[f"{comp}_ghia"]
            for k, p in enumerate(pts):
                if g is not None and _is_suspect(comp, r.Re, p):
                    xy = (g[k], prof["y"][k]) if comp == "u" else (prof["x"][k], g[k])
                    ax.annotate(
                        "Ghia: provável erro\nde impressão",
                        xy=xy,
                        xytext=(-95, -42),
                        textcoords="offset points",
                        fontsize=7.5,
                        color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=4),
                    )
            ax.grid(True)
            ax.set_axisbelow(True)
            ax.set_title(_case_label(r), fontsize=9)
            if r.status != "convergiu":
                ax.text(
                    0.03, 0.03, STATUS_PT[r.status], transform=ax.transAxes, fontsize=8, color=INK2,
                    va="bottom", ha="left",
                )
        for ax in axs[len(ordered):]:
            ax.axis("off")
        xlabel, ylabel, title = (
            ("u (x = 0.5)", "y", "Componente u ao longo da linha vertical x = 0.5")
            if comp == "u"
            else ("x", "v (y = 0.5)", "Componente v ao longo da linha horizontal y = 0.5")
        )
        for k, ax in enumerate(axs[: len(ordered)]):
            if k + 4 >= len(ordered):  # painel mais baixo da sua coluna
                ax.tick_params(labelbottom=True)
                ax.set_xlabel(xlabel)
            if k % 4 == 0:
                ax.set_ylabel(ylabel)
        fig.suptitle(title, color=INK, fontsize=11)
        handles, labels = axs[0].get_legend_handles_labels()
        if len(ordered) < len(axs):
            axs[-1].legend(handles, labels, loc="center", fontsize=10)
        else:
            fig.legend(handles, labels, loc="lower center", ncol=2)
        fig.tight_layout()
        path = os.path.join(folder, f"q2a_perfil_{comp}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _contour_panel(ax, x, y, f, levels, zero_line=False):
    neg = levels[levels < 0]
    pos = levels[levels > 0]
    if len(neg):
        ax.contour(x, y, f, levels=neg, colors=NEG_C, linewidths=0.7, linestyles="solid")
    if len(pos):
        ax.contour(x, y, f, levels=pos, colors=POS_C, linewidths=0.7, linestyles="solid")
    if zero_line:
        ax.contour(x, y, f, levels=[0.0], colors=ZERO_C, linewidths=0.7, linestyles="solid")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    for s in ax.spines.values():
        s.set_color(AXIS_C)


def plot_contours(results, folder=FIGS_DIR):
    """Item c: linhas de corrente (ψ) e vorticidade (ω) para os vários Re."""
    plt = _mpl()
    from matplotlib.lines import Line2D

    os.makedirs(folder, exist_ok=True)
    ordered = sorted(results.values(), key=lambda r: r.Re)
    paths = []
    for name, levels, title, zero in (
        ("psi", PSI_LEVELS, "Função corrente ψ", False),
        ("omega", OMEGA_LEVELS, "Vorticidade ω", True),
    ):
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.9))
        axs = axs.ravel()
        for ax, r in zip(axs, ordered):
            _contour_panel(ax, r.x, r.y, getattr(r, name), levels, zero_line=zero)
            ax.set_title(_case_label(r), fontsize=8.5)
        for ax in axs[len(ordered):]:
            ax.axis("off")
        if name == "psi":
            handles = [
                Line2D([], [], color=NEG_C, lw=1.5, label="ψ < 0 (sentido horário)"),
                Line2D([], [], color=POS_C, lw=1.5, label="ψ > 0 (anti-horário)"),
            ]
            note = "níveis de Ghia et al. (1982):\nψ = −0.1175 … −1e−10 e 1e−8 … 3e−3"
        else:
            handles = [
                Line2D([], [], color=NEG_C, lw=1.5, label="ω < 0"),
                Line2D([], [], color=ZERO_C, lw=1.5, label="ω = 0"),
                Line2D([], [], color=POS_C, lw=1.5, label="ω > 0"),
            ]
            note = "níveis: ω = 0, ±0.5, ±1, ±2, ±3, ±4, ±5"
        target = axs[-1] if len(ordered) < len(axs) else None
        if target is not None:
            target.legend(handles=handles, loc="center", fontsize=10, title=note, title_fontsize=8.5)
        else:
            fig.legend(handles=handles, loc="lower center", ncol=len(handles))
        fig.suptitle(title, color=INK, fontsize=11)
        fig.tight_layout()
        path = os.path.join(folder, f"q2c_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_residuals(results, folder=FIGS_DIR, name="q2_residuos.png"):
    """Histórico de convergência: ‖R_ψ‖∞ e ‖R_ω‖∞ ao longo das iterações."""
    plt = _mpl()
    os.makedirs(folder, exist_ok=True)
    ordered = sorted(results.values(), key=lambda r: (r.Re, r.N))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for k, r in enumerate(ordered):
        hst = r.history
        for ax, col in zip(axs, (1, 2)):
            ax.loglog(hst[:, 0], hst[:, col], color=SERIES[k % len(SERIES)], lw=1.3, label=_case_label(r))
    for ax, ttl in zip(axs, ("‖R_ψ‖∞", "‖R_ω‖∞")):
        for tol in sorted({r.tol for r in ordered}):
            ax.axhline(tol, color=MUTED, lw=0.9)
            ax.text(ax.get_xlim()[0] * 1.3, tol * 1.6, f"tol = {tol:g}", color=INK2, fontsize=8)
        ax.set_title(ttl)
        ax.set_xlabel("iteração")
        ax.grid(True, which="major")
        ax.set_axisbelow(True)
    axs[0].set_ylabel("resíduo (norma do máximo)")
    axs[1].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    path = os.path.join(folder, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_re1000(results, folder=FIGS_DIR):
    """Item b: |ψ|max × h para Re = 1000 (ADI, slide 23 e referências)."""
    plt = _mpl()
    os.makedirs(folder, exist_ok=True)
    rows = sorted((r for r in results.values() if np.isclose(r.Re, 1000.0)), key=lambda r: r.N)
    if not rows:
        return None
    ns = np.array([r.N for r in rows])
    ours = np.array([abs(primary_vortex(r)["psi"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    slide = sorted((int(k.split("x")[0]), v["psi"]) for k, v in TABLE_RE1000.items())
    ax.plot(
        *zip(*slide), "s", ms=11, mfc="none", mec=SERIES[1], mew=1.5, label="ADI do professor (slide 23)"
    )
    ax.plot(ns, ours, "-o", color=SERIES[0], lw=1.5, ms=5, mec="white", mew=0.8, label="ADI (este trabalho)")
    for name, ref in PRIMARY_VORTEX_RE1000.items():
        if "Ghia" in name or "Botella" in name:
            above = "Botella" in name
            ax.axhline(ref["psi"], color=MUTED, lw=0.9)
            ax.text(
                30,
                ref["psi"] + (0.0007 if above else -0.0007),
                f"{name}: {ref['psi']}",
                fontsize=7.5,
                color=INK2,
                va="bottom" if above else "top",
            )
    allns = sorted(set(ns) | {n for n, _ in slide})
    ax.set_xscale("log", base=2)
    ax.set_xticks(allns)
    ax.set_xticklabels([f"{n}×{n}" for n in allns])
    ax.minorticks_off()
    ax.set_xlabel("malha")
    ax.set_ylabel("|ψ| máximo")
    ax.set_title("Re = 1000: intensidade do vórtice primário")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(folder, "q2b_psi_max.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def animate_case(res, folder=ANIM_DIR, fps=12, gif=True, mp4=True):
    """Animação da evolução (pseudo-temporal) de ψ e ω a partir dos quadros salvos."""
    plt = _mpl()
    from matplotlib import animation

    fr = res.frames
    if fr is None or len(fr["iteracao"]) == 0:
        return []
    os.makedirs(folder, exist_ok=True)
    stride = max(1, res.N // 128)
    x, y = res.x[::stride], res.y[::stride]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.8))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.86, wspace=0.15)
    title = fig.suptitle("", color=INK, fontsize=10)

    def draw(k):
        for ax in axs:
            ax.clear()
        _contour_panel(axs[0], x, y, fr["psi"][k], PSI_LEVELS)
        _contour_panel(axs[1], x, y, fr["omega"][k], OMEGA_LEVELS, zero_line=True)
        axs[0].set_title("função corrente ψ", fontsize=9)
        axs[1].set_title("vorticidade ω", fontsize=9)
        it = int(fr["iteracao"][k])
        t = it * res.dt
        title.set_text(
            f"Re = {res.Re:g}, malha {res.N}×{res.N}, Δt = {dt_label(res.sigma)}  |  iteração {it}, "
            f"t = {t:.4f} (Re·t = {res.Re * t:.1f}), resíduo = {fr['residuo'][k]:.1e}"
        )
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(fr["iteracao"]), blit=False)
    base = os.path.join(folder, f"q2_animacao_Re{res.Re:g}_N{res.N}")
    out = []
    has_ffmpeg = animation.writers.is_available("ffmpeg")
    if has_ffmpeg and (mp4 or gif):
        # renderiza uma vez em MP4; o GIF é convertido do MP4 (paleta otimizada, arquivo menor)
        anim.save(base + ".mp4", writer=animation.FFMpegWriter(fps=fps, bitrate=1600), dpi=100)
        if gif:
            import subprocess

            filtro = "fps={0},scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse".format(fps)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", base + ".mp4", "-vf", filtro, base + ".gif"],
                check=True,
            )
            out.append(base + ".gif")
        if mp4:
            out.append(base + ".mp4")
        else:
            os.remove(base + ".mp4")
    elif gif:
        anim.save(base + ".gif", writer=animation.PillowWriter(fps=fps), dpi=72)
        out.append(base + ".gif")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def postprocess(animations=True):
    ghia = load_cases(GHIA_CASES)
    re1000 = load_cases(RE1000_CASES)
    allres = {**ghia, **re1000}
    if ghia:
        print_parameters_table(ghia, "Itens a/c — simulações para comparação com Ghia et al.")
        print_profile_tables(ghia)
        write_profile_tables(ghia)
        plot_profiles(ghia)
        plot_contours(ghia)
        plot_residuals(ghia)
    if re1000:
        print_parameters_table(re1000, "Item b — simulações para Re = 1000")
        print_re1000_table(re1000)
        write_re1000_table(re1000)
        plot_re1000(re1000)
        plot_residuals(re1000, name="q2b_residuos.png")
    if allres:
        write_parameters_table(allres)
    if animations:
        for c in GHIA_CASES:
            p = result_path(c)
            if not os.path.exists(p):
                continue
            gif = os.path.join(ANIM_DIR, f"q2_animacao_Re{c.Re:g}_N{c.N}.gif")
            if os.path.exists(gif) and os.path.getmtime(gif) >= os.path.getmtime(p):
                print("animação já atualizada:", os.path.relpath(gif, ROOT))
                continue
            for f in animate_case(load_result(p, with_frames=True)):
                print("animação:", os.path.relpath(f, ROOT))
    print(f"\nTabelas em {os.path.relpath(TABLES_DIR, ROOT)}/, figuras em {os.path.relpath(FIGS_DIR, ROOT)}/")


def main():
    ap = argparse.ArgumentParser(description="Questão 2 — cavidade com tampa deslizante (ADI)")
    ap.add_argument("--jobs", type=int, default=1, help="casos rodando em paralelo (processos)")
    ap.add_argument("--only-post", action="store_true", help="não roda simulações; só gera saídas")
    ap.add_argument("--force", action="store_true", help="refaz simulações já salvas")
    ap.add_argument("--no-anim", action="store_true", help="não gera animações")
    ap.add_argument("--run", nargs=3, metavar=("RE", "N", "DT_H2"), help="roda um único caso e salva")
    ap.add_argument("--resume", action="store_true", help="com --run: continua um caso salvo não convergido")
    ap.add_argument("--max-iter", type=int, default=2_000_000)
    ap.add_argument("--tol", type=float, default=TOL, help=f"tolerância do critério de parada (padrão {TOL:g})")
    args = ap.parse_args()

    if args.run:
        c = Case(float(args.run[0]), int(args.run[1]), float(args.run[2]), args.max_iter, args.tol)
        _run_and_save(c, resume=args.resume)
        return
    if not args.only_post:
        run_cases(GHIA_CASES + RE1000_CASES, jobs=args.jobs, force=args.force)
    postprocess(animations=not args.no_anim)


if __name__ == "__main__":
    main()
