"""ASTRA's published transport scheme, with Pereverzev and Corrigan's stabiliser.

ASTRA's source is not public, and its manual (Pereverzev & Yushmanov, IPP 5/98,
2002; `refs/Astra_ocr.pdf`) does not write down its difference scheme. What
*is* published is the scheme Pereverzev and Corrigan give as "the standard
form" in Comput. Phys. Commun. 179 (2008) 579 (`refs/PereverzevCorrigan.pdf`),
whose examples were run in ASTRA, together with the stabiliser ASTRA-8 (PPCF 68
(2026) 065024, Sec. 5.4) says nearly every stiff ASTRA study of the last two
decades has relied on. That is what this module implements, and it should be
called "the published ASTRA scheme", not ASTRA.

**Space** (PC eqs. (2)-(3)): a cell-centred grid, x_i = (i - 1/2) h, with fluxes
on the cell faces. Each channel's flux is sigma = D u_x + V u, the diffusivity D
and pinch V coming from the transport model; the face flux is Patankar's
exponential scheme, exact for constant D and V, and plain central differencing
when V = 0. Conservative, O(h^2).

**Time** (PC eq. (2)): backward Euler in u, with D, V and the sources evaluated
at the *old* time -- a lagged-coefficient linear step, one tridiagonal solve
per channel. Channels are coupled only through the lagged coefficients. A
source is split into a part linear in the channel's own profile, taken
implicitly, and the rest, explicitly (manual pp. 41-42).

**Nonlinearity**: none, beyond the lag. ASTRA's outer-iteration count `ITEREX`
(manual Table 4.15) defaults to 1; `iterex > 1` re-evaluates the coefficients
at the new iterate and re-solves -- a Picard iteration, which is the most the
manual's "No. of iterations in the external loop" can be taken to mean.

**Stabiliser** (PC eq. (14)): an added diffusive flux taken implicitly and the
same flux taken explicitly,

    sigma += Dbar (u_x^{new} - u_x^{old}),

which vanishes at a steady state and adds O(tau Dbar u_xt) in a transient.
Dbar is a diffusivity: ASTRA's equation carries the geometry V' outside the
diffusivities, so the added flux is V' Dbar u_x. (Added to V' D instead, a
constant Dbar swamps a diffusivity that vanishes like x on the axis, and the
iteration near the axis slows in proportion to the number of cells.) PC's
eq. (8) form, a compensating pinch Dbar u_x/u, is equivalent to within O(h^2)
and needs u != 0; Jardin's problem has u(1) = 0, so eq. (14) it is. Dbar is a
constant per problem, chosen above the slope of the flux-gradient curve.

**Time step** (PC eq. (19)): after each step the model is evaluated at the new
state, and

    Delta_D = max_i |D_new - D_old| / D_old

is compared with a tolerance, 10% in PC's runs. Above it, the step is repeated
from the old state with tau halved; below it, the step is accepted and tau
multiplied by TAUINC = 1.1, ASTRA's growth factor (manual Table 4.15). The
evaluation at the new state is the one the next step needs anyway, so an
accepted step costs one model evaluation per face and a rejected one wastes
one. The halving factor is not published; 1/2 is a choice.

**Stopping**: ASTRA marches to a time. Here the march stops when the discrete
steady residual at the accepted state, with that state's coefficients, falls
below `tol` relative to the size of its terms -- computed from the evaluation
the step control already made, so it costs nothing.
"""

import numpy as np
from scipy.linalg import solve_banded


def _alpha(xi):
    """Patankar's alpha(xi) = xi / (1 - exp(-xi)), with the removable
    singularity at xi = 0 handled."""
    xi = np.asarray(xi, dtype=float)
    small = np.abs(xi) < 1e-8
    safe = np.where(small, 1.0, xi)
    return np.where(small, 1.0 + 0.5 * xi, safe / -np.expm1(-safe))


class Result:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _faces(problem, U, h):
    """Face positions, profiles and gradients for the N faces off the axis.
    Interior faces take the average and the difference of their two cells; the
    wall face takes the Dirichlet value and the half-cell difference."""
    nv, N = U.shape
    xf = h * np.arange(1, N + 1)
    Uf = np.empty((nv, N))
    Qf = np.empty((nv, N))
    Uf[:, :-1] = 0.5 * (U[:, :-1] + U[:, 1:])
    Qf[:, :-1] = (U[:, 1:] - U[:, :-1]) / h
    Uf[:, -1] = problem.ub
    Qf[:, -1] = (problem.ub - U[:, -1]) / (0.5 * h)
    return xf, Uf, Qf


def _face_weights(D, V, h):
    """Coefficients of the face flux sigma_f = D (alpha u_R - beta u_L) / d,
    d being h inside and h/2 at the wall. Returns (aR, bL): sigma = aR u_R -
    bL u_L. PC write the flux as q = V' u - D u_x with V' = -V; their Peclet
    number xi = -h V'/D is h V/D here."""
    d = np.full(D.shape[-1], h)
    d[-1] = 0.5 * h
    xi = d * V / D
    a = _alpha(xi)
    return D * a / d, D * (a - xi) / d


def _step(problem, U, D, V, S0, S1, tau, dbar, h):
    """One lagged backward-Euler step for every channel."""
    nv, N = U.shape
    W = np.empty_like(U)
    # The stabiliser's flux on each face, geometry included.
    Dbar = dbar * problem.geometry(h * np.arange(1, N + 1))
    for s in range(nv):
        aR, bL = _face_weights(D[s], V[s], h)
        d = np.full(N, h)
        d[-1] = 0.5 * h
        aR = aR + Dbar / d
        bL = bL + Dbar / d
        # The explicit half of the stabiliser, per face.
        du = np.empty(N)
        du[:-1] = U[s, 1:] - U[s, :-1]
        du[-1] = problem.ub[s] - U[s, -1]
        c = -Dbar * du / d
        c[-1] += aR[-1] * problem.ub[s]       # the wall face's known u_R
        # Face j (1..N) is entry j-1 of these arrays; cell i lies between faces
        # i (left, absent on the axis) and i+1 (right).
        diag = 1.0 / tau - S1[s] + bL / h
        diag[1:] += aR[:-1] / h
        upper = -aR[:-1] / h                  # coefficient of u_{i+1} in row i
        lower = -bL[:-1] / h                  # coefficient of u_{i-1} in row i+1
        rhs = U[s] / tau + S0[s] + c / h
        rhs[1:] -= c[:-1] / h
        ab = np.zeros((3, N))
        ab[0, 1:] = upper
        ab[1] = diag
        ab[2, :-1] = lower
        W[s] = solve_banded((1, 1), ab, rhs)
    return W


def _steady_residual(problem, U, D, V, h):
    """max |d_x sigma + S| over the max size of its terms, at U with U's own
    coefficients and no stabiliser."""
    nv, N = U.shape
    xc = h * (np.arange(N) + 0.5)
    S0, S1 = problem.source(xc, U)
    S = S0 + S1 * U
    worst, scale = 0.0, 0.0
    for s in range(nv):
        aR, bL = _face_weights(D[s], V[s], h)
        F = np.empty(N)
        F[:-1] = aR[:-1] * U[s, 1:] - bL[:-1] * U[s, :-1]
        F[-1] = aR[-1] * problem.ub[s] - bL[-1] * U[s, -1]
        div = (F - np.concatenate(([0.0], F[:-1]))) / h
        worst = max(worst, np.max(np.abs(div + S[s])))
        scale = max(scale, np.max(np.abs(div)), np.max(np.abs(S[s])))
    return worst / scale if scale > 0 else worst


def solve(problem, ncells, dbar=None, tau0=1e-3, dtol=0.1, shrink=0.5, tauinc=1.1,
          iterex=1, tol=1e-10, max_steps=200000, tau_min=1e-14):
    """March `problem` to its steady state on `ncells` cells. Returns a Result
    with the cell values, a callable profile, the model evaluations spent and
    the step counts."""
    if dbar is None:
        dbar = problem.dbar
    h = 1.0 / ncells
    xc = h * (np.arange(ncells) + 0.5)
    U = problem.initial(xc)
    start = problem.evals
    D, V = problem.coefficients(*_faces(problem, U, h))
    tau, t = tau0, 0.0
    accepted = rejected = 0
    converged = False
    res = np.inf
    while accepted + rejected < max_steps:
        S0, S1 = problem.source(xc, U)
        W = _step(problem, U, D, V, S0, S1, tau, dbar, h)
        for _ in range(iterex - 1):
            Di, Vi = problem.coefficients(*_faces(problem, W, h))
            S0, S1 = problem.source(xc, W)
            W = _step(problem, U, Di, Vi, S0, S1, tau, dbar, h)
        Dn, Vn = problem.coefficients(*_faces(problem, W, h))
        with np.errstate(divide="ignore", invalid="ignore"):
            change = np.where(D > 0, np.abs(Dn - D) / D, 0.0)
        if not np.all(np.isfinite(W)) or np.max(change) > dtol:
            rejected += 1
            tau *= shrink
            if tau < tau_min:
                break
            continue
        accepted += 1
        t += tau
        U, D, V = W, Dn, Vn
        tau *= tauinc
        res = _steady_residual(problem, U, D, V, h)
        if res < tol:
            converged = True
            break

    ub = problem.ub

    def profile(x):
        x = np.asarray(x, dtype=float)
        # Linear through the cell centres, the Dirichlet value at the wall, and
        # extrapolated linearly from the first two centres to the axis.
        axis = U[:, :1] - 0.5 * (U[:, 1:2] - U[:, :1])
        xs = np.concatenate(([0.0], xc, [1.0]))
        out = []
        for s in range(U.shape[0]):
            out.append(np.interp(x, xs, np.concatenate((axis[s], U[s], [ub[s]]))))
        return np.array(out)

    return Result(U=U, profile=profile, evals=problem.evals - start, accepted=accepted,
                  rejected=rejected, t=t, converged=converged, residual=res,
                  points=ncells)
