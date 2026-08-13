# Shestakov's degenerate-diffusion benchmark

Nonlinear diffusion whose diffusivity is the inverse-square density scale
length, so the flux goes as the gradient *cubed*. From Shestakov, Cohen,
Crotinger, LoDestro, Tarditi & Xu, *Self-consistent modeling of turbulence and
transport*, Journal of Computational Physics **185** (2003) 399–426, Section
2.1, with the corrigendum at JCP **186** (2003) 360
([`refs/`](../../refs/Refs.md)).

The third of the three benchmarks in this tree, and the one that ends
differently. [`../park-convergence/`](../park-convergence/) measures spatial
accuracy per call and [`../jardin-critical-gradient/`](../jardin-critical-gradient/)
measures the nonlinear solve; **this one marks a boundary MaNTA's formulation
cannot currently cross**, and identifies what stops it.

## The problem

    d_t n + d_x[ Gamma ] = S,   Gamma = -D n_x,   D = l_n^-2 = (n_x/n)^2

so in the form MaNTA integrates, `a d_t u - d_x[sigma_hat] = S`,

    sigma_hat = D0 q^3 / u^2,   q = du/dx

    S(x) = S0 for x < d, 0 otherwise      (S0 = 1, d = 0.1)
    sigma_hat(0) = 0                      (Neumann)
    n(Lx) = n_b                           (Dirichlet; Lx = 1)

Shestakov uses it to show a *semi-implicit* scheme — one that lags `D` — going
unstable. Here `D ~ (d_x n)^p n^q` with **p = 2, q = −2**, and his analysis
gives instability for `p > 1` once `n^2 D dt > 2/(p−1)`. Neither of the
corrigendum's caveats applies: its short-wavelength instability of the *fully*
implicit scheme needs `−2 < n^2 dt D (p+1) < 0` and `p+1 = 3 > 0` here, and its
singular-matrix warning needs `D` to change sign, which a square cannot.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/shestakov-nonlinear
    manta run.conf           # 10 cells, k = 2, n(Lx) = 0.1
    python benchmark.py      # cost, the tractability wall, and the diagnosis

## Two misprints in the paper

**The steady-state branches are swapped.** Integrating `d_x Gamma = S` from the
zero-flux axis gives `Gamma = S0 min(x, d)`, and `w = n^(1/3)` linearises the
rest to `w_x = -(1/3)(Gamma/D0)^(1/3)`, so

    n_e(x) = [ n_b^(1/3) + (1/3)(S0 d/D0)^(1/3) (Lx - x) ]^3               x >= d
    n_e(x) = [ n_b^(1/3) + (1/3)(S0/D0)^(1/3)(
                   0.75(d^(4/3) - x^(4/3)) + d^(1/3)(Lx - d)) ]^3          x <  d

The paper's brace labels `(Lx − x)^3` as the `x < d` branch. Substituted into
the steady equation that leaves a residual of exactly `−S`; swapped, the
residual is finite-difference noise (~1e-6). The `S0 d/27 D0` prefactor
multiplies **both** branches, and the pair above reduces to the paper's at
`n_b = 0`. Continuity at `x = d` and both boundary conditions check out.

**The similarity inversion drops a factor.** The paper gives
`eta = x^4/(64 D0 t)` and then inverts it as `x_f = eta_f t^(1/4)`, which is
missing `(64 D0)^(1/4)`. With the factor restored, the printed `eta_f = 3.339`
reproduces the paper's own tabulated front positions — 0.785, 0.618, 0.321 — to
three figures; as printed it is out by 3–8%. So `eta_f` is right and the
inversion is misprinted.

## Where MaNTA gives out

Shestakov's Section 2.1 sets `n_b = 0`. **MaNTA cannot integrate that at any
resolution or tolerance tried.** `benchmark.py` maps the wall, at 10 cells and
k = 2:

| `n(Lx)` | 0.2 | 0.1 | 0.07 | 0.05 | 0.03 | 0.01 | 1e-3 | 0 |
|---|---|---|---|---|---|---|---|---|
| | ok | ok | ok | fail | fail | fail | fail | fail |

Two separate obstructions, and neither is the nonlinear solve.

**The paper's initial condition is unusable.** `n0 ≡ 1` contradicts the Dirichlet
value at `x = Lx`, and being constant it makes `D ≡ 0` and `d(sigma_hat)/dq ≡ 0`
everywhere, so the trace system degenerates and `IDACalcIC` fails outright.
Shestakov's primal FEM absorbs both — his mass matrix is invertible with `D = 0`,
and he lags `D` so the first step simply resolves the mismatch — where a DAE
with algebraic trace rows cannot start there. This case therefore keeps his
`n(0) = 1` and zero axis flux but meets the Dirichlet value and vanishes like
`(Lx − x)^3`, so the flux stays finite at the wall as the true solution does.

**The flux is unbounded as the wall density falls.** `sigma_hat = D0 q^3/u^2` is
a 0/0 at `u = 0` — finite in the exact solution, where `n ~ (Lx−x)^3` and
`q ~ (Lx−x)^2` — but unbounded in a perturbed one, and the Jacobian entry
`d(sigma_hat)/du = -2 D0 q^3/u^3` diverges outright. Shestakov never forms
either derivative: he lags `D` over previous iterates and needs no Jacobian at
all, and his `max(0, D)` plus lumping give him an M-matrix that *guarantees*
`n >= 0`. MaNTA has no positivity mechanism and a Newton method that must
evaluate the derivative that blows up. This problem probes exactly that gap.

**What actually fails is IDA's error test.** Run under `SUNLOGGER_INFO_FILENAME`,
every failing step shows the Newton converging in **2 iterations, status
success**, and the step then dying with `dsm = 15.7011094228028` — *identical*
at `h = 1e-7` and at larger `h`. A local error estimate that will not shrink
with `h` is the signature of an algebraic component in the error test, and
`IDASetSuppressAlg` is never called, so `sigma` is in it. Since `sigma` is what
grows without bound here, the error test is unsatisfiable however small the step.

That diagnosis predicts a cure, and it works: at the paper's Section 2.2 value
`n_b = 1e-3`, loosening `Absolute_tolerance` alone admits the problem —

| `atol` | 1e-3 | 1e-1 | **1e0** | 1e1 | 1e2 |
|---|---|---|---|---|---|
| | fail | fail | **ok, err 3.3e-2** | fail | fail |

— and only within a window, because too loose lets the solution wander to where
the flux blows up anyway. This is the same `Absolute_tolerance` sensitivity the
other two benchmarks note, but here it is coupled to the physics rather than
being a free choice.

## Cost where it is tractable

At `n(Lx) = 0.1`, in PERFORMANCE.md's units:

| cells | k | points | flux calls | deriv pts | visits/point | error |
|---|---|---|---|---|---|---|
| 10 | 1 | 20 | 5380 | 780 | 308 | 2.55e-2 |
| 10 | 2 | 30 | 9420 | 1200 | 354 | 1.06e-2 |
| 10 | 3 | 40 | 12800 | 1760 | 364 | 5.87e-3 |
| 20 | 2 | 60 | 19140 | 2760 | 365 | 5.21e-3 |
| 20 | 3 | 80 | 26160 | 3760 | 374 | 2.90e-3 |

Two things to read here. The **visit count is 300–375**, against ~120 for Park's
problem and ~200 for Jardin's — this is much the hardest of the three for the
time integrator, which is the point of it.

And the **error is ~1e-2 and falls slowly, which is the solution's regularity
rather than the scheme**. `n_e` carries an `x^(4/3)` at the axis, whose second
derivative is unbounded, and the step source puts a kink at `x = d`. `Grid_size`
is 10 in `run.conf` so a cell boundary falls on that kink; nothing can be done
about the axis. An order study on this problem will not measure what it looks
like it measures.

## What would close the gap

Nothing here is a defect to be fixed by tuning. Getting `n_b = 0` would need one
of the things Shestakov's scheme has and MaNTA does not — a positivity-preserving
discretisation, a lagged-diffusivity iteration that never differentiates the
flux, or `IDASetSuppressAlg` so the algebraic rows leave the error test. The
last is a one-line experiment and is the obvious first thing to try.
