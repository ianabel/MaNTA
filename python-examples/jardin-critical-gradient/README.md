# Jardin's critical-gradient benchmark

The stiff problem that motivates every algorithm in [`refs/`](../../refs/Refs.md):
1-D diffusion whose diffusivity depends strongly and non-analytically on the
*gradient*. From Jardin, Bateman, Hammett & Ku, *On 1D diffusion problems with a
gradient-dependent diffusion coefficient*, Journal of Computational Physics
**227** (2008) 8769–8775 — filed as `refs/PTRANSP.pdf`, though despite the file
name that paper is not about PTRANSP.

`PERFORMANCE.md` says MaNTA's performance should be measured by the number of
calls into a `TransportSystem` needed for a given accuracy, and compared against
the algorithms in `refs/`. This is that comparison for the **nonlinear solve**;
[`../park-convergence/`](../park-convergence/) is the same exercise for
**spatial accuracy**.

## The problem

    d_t u - d_x[ x chi(q) q ] = 1,     q = du/dx

    chi(q) = chi0 + kappa (|q| - qc)^alpha    for |q| > qc
           = chi0                             otherwise

with Jardin's `chi0 = 1, kappa = 10, alpha = 0.5, qc = 0.5`, `u(1) = 0`, his
initial condition `u = 1 - x` (the steady state of the same problem with `chi`
held at `chi0`), and on the axis — read the next section before assuming — a
Neumann value of `-g`, not zero.

This is the shape that defeats a plain implicit time step. Jardin shows backward
Euler oscillating and needing a step some **four orders of magnitude smaller**
than his linearised scheme to reach the right answer, and Shestakov gives the
matching stability analysis: for `D ~ (du/dx)^p`, lagging `D` is unstable for
`p < -1` at *any* time step.

**The stiff steady state is exactly linear.** Integrating once and requiring
regularity on the axis gives `chi(q) q = -1`, so `q` is a constant `-g` with

    [chi0 + kappa (g - qc)^alpha] g = 1,    g = 0.5092841043...

and `u = g (1 - x)`. Being degree 1 it lies in `P_k` at every order this solver
runs at, which is what takes the spatial discretisation out of the answer and
leaves the nonlinear solve on its own.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/jardin-critical-gradient
    manta run.conf           # one run at 10 cells, k = 3
    python benchmark.py      # the robustness-and-cost study

`manta run.conf` writes `run.nc` and `run.restart.nc` beside the config.
`benchmark.py` writes nothing and prints three tables; the solver's own progress
lines interleave, so the tables are collected and printed at the end.

## What it measures

MaNTA reproduces the stiff steady state **to round-off at every resolution**,
which is the right answer rather than an impressive one: `u = g(1-x)` is degree
1 and lies in `P_k` for every `k` here, so any correct scheme must give it
exactly. The table therefore measures cost alone:

| cells | k | points | flux calls | deriv pts | visits/point | error |
|---|---|---|---|---|---|---|
| 4 | 2 | 12 | 1896 | 456 | 196 | 3.0e-16 |
| 4 | 3 | 16 | 3088 | 576 | 229 | 4.7e-16 |
| 10 | 3 | 40 | 7160 | 1640 | 220 | 8.0e-16 |
| 10 | 5 | 60 | 11700 | 2400 | 235 | 1.1e-15 |

For comparison, on a problem of this kind:

* **Jardin** — 1–3 Newton iterations per step, each needing `chi` and
  `dchi/dq`; PTRANSP uses 3. Tridiagonal, no assembled Jacobian.
* **Park** — 9–15 root-finding iterations for a steady state, one
  transport-model call per grid point per iteration, and no derivatives at all.
* **TRINITY** — `n_r x (1 + n_p)` flux evaluations per transport step, 2 Newton
  iterations per step, 10–15 steps.

**Where MaNTA wins outright is the Jacobian.** TRINITY finite-differences its
flux Jacobian, so it pays a `(1 + n_p)` multiplier on the expensive model — 4×
for its three evolved profiles. Jardin's whole contribution is avoiding that
with a semi-analytic linearisation that keeps the matrix tridiagonal. MaNTA goes
further: `dSigmaFn_dq` returns the tangent diffusivity `x [chi + (dchi/dq) q]`
analytically, so the `deriv pts` column above costs **no flux evaluations at
all**. Supply that hook from `autodiff` or `jax.grad` and it stays free.

**Where it loses is the same place as the other benchmark** — the `visits`
column — and `SteadyStateSolver` closes some of it. Time marching needs 212
visits per point here; pseudo-transient continuation needs **127** and Newton
152, for the same round-off answer. Less of a win than Park's 113 → 11, because
this problem's stiffness is real rather than an artefact of error control, but
still most of a factor of two. See `docs/running.rst`.

## The trap: a Neumann boundary fixes q, not the flux

This example shipped with the axis boundary value at zero, on the reasoning that
the physical condition there is zero flux. That reasoning is wrong twice over,
and it cost a factor of `1e12` in accuracy.

**MaNTA's Neumann boundary fixes `q`, the gradient — not the flux.** The two
coincide only when `sigma_hat = q`, which is every other case in this tree, so
nothing had ever exercised the difference. Measured directly: with
`sigma_hat = 2q` and a Neumann value of `0.3`, the converged solution has
`q = 0.3` and `sigma_hat = 0.6`.

**And Jardin's problem has no condition on the axis at all.** `sigma_hat = x
chi(q) q` vanishes at `x = 0` for *any* `q`, so the flux condition is satisfied
identically and regularity alone selects the solution. Asking for a zero Neumann
value does not express that; it imposes `q(0) = 0`, which is an extra constraint,
and a false one — the true gradient on the axis is `-g = -0.509`.

What that looked like, before the cause was known:

| cells | k | error, `q(0) = 0` | error, `q(0) = -g` |
|---|---|---|---|
| 4 | 2 | 5.78e-3 | 3.0e-16 |
| 10 | 3 | 7.49e-4 | 8.0e-16 |
| 20 | 3 | 3.33e-4 | — |
| 40 | 3 | 1.58e-4 | — |

Exactly first order in `h` and **independent of `k`** — `k = 5` bought no rate
at all over `k = 3`. That combination is the tell: a genuine discretisation
limit improves with order, and a wrong boundary condition does not. The error
was a one-cell layer on the axis where `q` was dragged from `-0.509` to `0`,
which also explains why it sat in `chi`'s flat branch there (`|q| < qc`) and why
the layer's width tracked `h`.

It also explains a second symptom that had been recorded as separate and
unexplained: starting the run *from* the exact steady state used to fail in
`IDACalcIC` with `IDA_CONV_FAIL`. Of course it did — `q = -g` everywhere
directly contradicts the imposed `q(0) = 0`, so there is no consistent initial
state to find. With the boundary value corrected, both starting points converge
to round-off.

Two things worth taking from this beyond the one example. `docs/physics_interface.rst`
said "a Neumann boundary fixes the flux" until this was measured, and now
carries the correction and a warning. And a problem whose flux vanishes at a
boundary *by degeneracy* cannot be posed to MaNTA as a zero Neumann value at
all: the correct gradient there is whatever regularity demands, which in general
you do not know without solving the problem. Here it is `-g`, which the closed
form supplies; `../park-convergence/` and `../shestakov-nonlinear/` both
genuinely have `q = 0` on the axis, so both were right by luck of the physics
rather than by reasoning.

## A limiter MaNTA does not have

Jardin's implementation notes warn that the tangent diffusivity
`chi + (dchi/dq) q` — precisely what `dSigmaFn_dq` returns, divided by `x` — can
go **negative**, "such as could happen at a transport barrier bifurcation",
producing "a severe numerical instability"; they had to clamp negative
`dchi/dq` to keep it positive. Shestakov hits the same wall from the other side
and clamps the *secant* diffusivity `-Gamma/grad n` with a `max(0, ·)` to keep
his matrix an M-matrix.

MaNTA has neither guard, and this benchmark does not need one: at its steady
state the tangent diffusivity is about `28x`, comfortably positive. A case that
wanders into the negative region is on its own, and would be the first in the
tree to care.
