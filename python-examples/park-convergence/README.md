# Park's convergence benchmark

A steady cylindrical transport problem with a closed-form solution, taken from
Park et al., *An efficient transport solver for tokamak plasmas*, Computer
Physics Communications **214** (2017) 1–5, Section 3
([`refs/ParkEfficientSolver.pdf`](../../refs/Refs.md)).

`PERFORMANCE.md` says MaNTA's performance should be measured by the number of
calls into a `TransportSystem` needed for a given accuracy, and compared against
the algorithms in `refs/`. This is that comparison for **spatial accuracy**;
[`../jardin-critical-gradient/`](../jardin-critical-gradient/) is the same
exercise for the **nonlinear solve**.

## The problem

Writing `V' = x` for a cylinder and taking `n = chi = 1`, Park's Eq. (1) in the
form MaNTA integrates is

    d_t u - d_x[ x q ] = S,   q = du/dx,   S = 4 x (1 - x^2) exp(1 - x^2)

with `sigma_hat(0) = 0` on the axis and `u(1) = 0`, whose steady state is

    u(x) = exp(1 - x^2) - 1

`chi` is constant, so the nonlinear iteration that every scheme in `refs/` is
built around does no work — which is the point. This benchmark isolates
accuracy per call from the nonlinear machinery.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/park-convergence
    manta run.conf           # one run at 4 cells, k = 4
    python benchmark.py      # the accuracy-against-cost study

`manta run.conf` writes `run.nc` and `run.restart.nc` beside the config.
`benchmark.py` writes nothing; it prints two tables and takes a couple of
minutes. The solver prints its own progress lines as it goes, so the tables are
collected and printed at the end — pipe through `tail -40` if that is in the way.

## What it measures

Park reports that his 4th-order IDO scheme at `N = 11` grid points is as
accurate as a 2nd-order finite difference at `N = 101`. His figure carries no
readable numbers, so `benchmark.py` implements that second-order scheme — his
Eq. (17) — and uses it as a bridge, which puts Park's `N = 11` result at a
relative L1 error of about `6e-5`. **That is a proxy for his published number,
not his published number.**

Measured, in his error norm:

| scheme | points | visits/point | flux evals | error |
|---|---|---|---|---|
| 2nd-order FD, N = 101 | 101 | 1 | 101 | 6.02e-5 |
| Park IDO, N = 11 | 11 | 1 | ~11 | ≈6e-5 *(proxy)* |
| MaNTA, 4 cells, k = 3 | 16 | 119 | 1504 | 7.55e-5 |
| MaNTA, 4 cells, k = 4 | 20 | 118 | 1860 | 3.41e-6 |
| MaNTA, 4 cells, k = 5 | 24 | 124 | 2352 | 2.12e-7 |

Two things fall out, and they point in opposite directions.

**Per evaluation point MaNTA is in IDO's league**, and both are far ahead of
second-order finite differences: 16 points reach what IDO reaches in 11, and
what the FD scheme needs 101 for. That is the HDG discretisation earning its
keep at exactly the resolutions `PERFORMANCE.md` targets.

**MaNTA visits each point ~120 times where the others visit once.** That is the
whole of the difference, and it is structural rather than a tuning problem:
`TerminateOnSteadyState` is a stopping test on the time-marching loop
(`Solver.cpp:470`), so MaNTA *relaxes onto* the steady state. Park solves for it
directly by setting `1/dt = 0`; LoDestro sets `dt = 1e10`; TRINITY converges in
10–15 transport steps. Note this is the price of *steady-state* runs only — for
a genuine transient, adaptive BDF1–5 with error control is doing more work than
any of them, not less.

Also worth noting: every point on the cost/accuracy Pareto frontier used **4
cells**, the coarsest tried. Refining the mesh never paid; raising `k` always
did. That is `PERFORMANCE.md`'s low-cell-count, high-order stance, measured.

## Superconvergence is a real trade, and it depends on k

`benchmark.py`'s third table compares `Superconvergent` on and off *at matched
cost*, on the postprocessed `u*` — which is the fair comparison, since `u*` is
built either way and the flag controls only whether the method uses it.
The flag costs exactly the extra node, `(k+2)/(k+1)`:

| budget | flag on | flag off | winner |
|---|---|---|---|
| ~1850 | 4 cells, k = 3: 2.65e-5 | 4 cells, k = 4: **2.20e-6** | k, by 12× |
| ~2300 | 4 cells, k = 4: **1.11e-7** | 4 cells, k = 5: 1.73e-7 | flag, by 1.6× |
| ~2800 | 6 cells, k = 3: 5.02e-6 | 6 cells, k = 4: **2.00e-7** | k, by 25× |

So the honest answer is *it depends on `k`* — raising the order wins at `k = 2–3`
and the flag wins at `k >= 4`. This is a fixed-resolution cost measurement and
says nothing about orders of accuracy; for those see
`Tests/UnitTests/MMSConvergenceTests.cpp` and `Tests/README.md`.

## Two traps

**Park's printed source and solution disagree by a factor of four.** He gives
`S = S0 (1-rho^2) exp(1-rho^2)` against `T = S0/(n chi) (exp(1-rho^2) - 1)`;
substituting the solution into the steady equation gives `S = 4 S0 (...)`. Both
satisfy the stated boundary conditions, so nothing catches it. The consistent
pair is what is implemented here. Check this before porting his Figure 2
anywhere else.

**`Absolute_tolerance` is what breaks, not `Relative_tolerance`.** At or below
`1e-7` this problem fails on the first step with `IDA_ERR_FAIL` (-3) at every
resolution; at `1e-3` it runs happily with `Relative_tolerance` as tight as
`1e-10`. IDA never calls `IDASetSuppressAlg`, so `sigma`, `q` and `lambda` are
all in its error test. Nothing is lost by keeping it loose: the error here is
set by the mesh and is unchanged across four decades of `Relative_tolerance`.
