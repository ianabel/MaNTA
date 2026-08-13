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
    manta run.conf           # 10 cells, k = 2, n(Lx) = 0.01
    python benchmark.py      # cost, the tractability wall, and the diagnosis

`run.conf` sets `SuppressAlgebraicError = true`, without which this problem does
not run at all — see below, and `docs/running.rst` for what that key costs.

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

## Where MaNTA gives out, and what moves the wall

Shestakov's Section 2.1 sets `n_b = 0`, and **that remains out of reach**: it
runs at exactly one of the five resolutions tried, and only with
`SuppressAlgebraicError`. What that key does buy is most of the way there.
`benchmark.py`'s second table, relative L1 error against the closed form:

| `n(Lx)` | flag | 10c/k1 | 10c/k2 | 10c/k3 | 20c/k2 | 20c/k3 |
|---|---|---|---|---|---|---|
| 0 | off | fail | fail | fail | fail | fail |
| 0 | **on** | fail | 2.55e-2 | fail | fail | fail |
| 1e-3 | off | fail | fail | fail | fail | fail |
| 1e-3 | **on** | fail | 3.30e-2 | 1.85e-2 | 1.64e-2 | 5.32e-2 |
| 1e-2 | off | fail | fail | fail | fail | fail |
| 1e-2 | **on** | 4.66e-2 | 1.99e-2 | 1.11e-2 | 9.83e-3 | 5.52e-3 |
| 0.1 | off | 2.55e-2 | 1.06e-2 | 5.87e-3 | 5.21e-3 | 2.90e-3 |
| 0.1 | **on** | 2.55e-2 | 1.06e-2 | 5.87e-3 | 5.21e-3 | 2.90e-3 |

So `n_b >= 1e-2` goes from failing everywhere to working everywhere, the paper's
own Section 2.2 value of 1e-3 becomes usable at most resolutions, and Section
2.1's zero is widened into rather than cured. Note the last two rows: where both
settings work the flag changes the answer by **nothing at all** — every digit
printed agrees.

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

That diagnosis predicts two cures, and both work. Loosening `Absolute_tolerance`
alone admits the paper's Section 2.2 value, in a narrow window — at `n_b = 1e-3`,
`atol = 1e0` runs where `1e-1` and `1e1` do not, the upper end failing because
too loose lets the solution wander to where the flux blows up anyway.

The better one is to take the algebraic rows out of the test altogether, which
is what `SuppressAlgebraicError = true` does (`IDASetSuppressAlg`). `run.conf`
sets it, and at this case's `BOUNDARY_DENSITY` of 0.01 the problem does not run
at all without it. **It is off by default in MaNTA and should stay that way**:
`sigma`, `q`, `lambda` and `phi` are then controlled only by the Newton
tolerance, a restart file serialises all of them — a round trip degrades from
`1.9e-6` to `8.6e-4` — and `phi` is a physics quantity in its own right when
`nAux > 0`, where the `AuxVarTest` regression case drifts 1.0% past its 0.84%
tolerance. `docs/running.rst` has the full account.

## Cost where it is tractable

At `n(Lx) = 0.05` with the flag on, in PERFORMANCE.md's units:

| cells | k | points | flux calls | deriv pts | visits/point | error |
|---|---|---|---|---|---|---|
| 10 | 1 | 20 | 4040 | 660 | 235 | 3.06e-2 |
| 10 | 2 | 30 | 5580 | 900 | 216 | 1.29e-2 |
| 10 | 3 | 40 | 8880 | 1360 | 256 | 7.17e-3 |
| 20 | 2 | 60 | 12180 | 2040 | 237 | 6.35e-3 |
| 20 | 3 | 80 | 14480 | 2960 | 218 | 3.55e-3 |

Two things to read here. The **visit count is 216–256**, against ~120 for Park's
problem and ~200 for Jardin's — this is still the hardest of the three for the
time integrator, which is the point of it. It was 308–375 before
`SuppressAlgebraicError`: dropping the algebraic rows from the error test buys
roughly a third off the call count as well as admitting the problem at all,
which is the same 13–44% saving the other two benchmarks show.

And the **error is ~1e-2 and falls slowly, which is the solution's regularity
rather than the scheme**. `n_e` carries an `x^(4/3)` at the axis, whose second
derivative is unbounded, and the step source puts a kink at `x = d`. `Grid_size`
is 10 in `run.conf` so a cell boundary falls on that kink; nothing can be done
about the axis. An order study on this problem will not measure what it looks
like it measures.

## What would close the remaining gap

Of the three things Shestakov's scheme has and MaNTA does not, one has now been
tried. `IDASetSuppressAlg` is in the tree as `SuppressAlgebraicError`, and it
moved the wall from `n_b >= 0.07` to `n_b >= 1e-2` while making the runs a third
cheaper — but it did not reach `n_b = 0`, and it is not free, so it is off by
default.

The two that remain are the ones that would: a **positivity-preserving
discretisation**, which is what Shestakov's lumped M-matrix construction gives
him and what stops `u` ever reaching the region where `q^3/u^2` is meaningless;
or a **lagged-diffusivity iteration** that never differentiates the flux at all,
which is what makes his scheme indifferent to a Jacobian entry that diverges.
Neither is a small change, and neither is a defect to be fixed by tuning.
