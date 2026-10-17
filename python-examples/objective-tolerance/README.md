# Stopping on the objective

`SteadyStateTest = "Objective"` stops a steady solve when the outputs a study
asks for are converged, rather than the whole solution. This measures it on the
four deterministic benchmarks beside this directory, against `NewtonStep` at its
default and against the discrete fixed point.

The question it answers is the one a transport study actually has: "the stored
energy to within 1%". `NewtonStep` answers a different one — every component of
the solution to within the run's `Relative_tolerance` and `Absolute_tolerance` —
and how that translates into the error in a stored energy is nowhere written
down. Here it is the tolerance.

## The test

For an objective `G = int g dx`, the solve stops in its undamped Newton solve
once, for every objective,

    |z^T F|  <=  share * max( ObjectiveTolerance |G|, ObjectiveAbsoluteTolerance )

where `z` is the adjoint state, `J^T z = dG/dy`, and `share` is the algebraic
half of `ObjectiveBudget`. `z^T F` is the dual-weighted estimate of how far `G`
at the iterate is from its value at the fixed point. It is not solved for: with
`J^T z = dG/dy` it equals `(dG/dy) J^-1 F`, the change in `G` that the Newton
correction the iteration has just computed makes, so the test costs the
objective's `dg` and a dot product per iteration and asks the physics for
nothing. The damped continuation before the undamped solve is `NewtonStep`'s,
step for step. `docs/running.rst` has the rest.

## Running it

    pip install .            # from the repository root, once
    cd python-examples/objective-tolerance
    python benchmark.py

It writes nothing and takes under a minute. The solver prints its own progress;
the table is collected and printed at the end.

## The problems and their objectives

| problem | resolution | objective |
|---|---|---|
| [Park](../park-convergence/) | 4 cells, k = 3 | `W = int x u dx` |
| [Jardin](../jardin-critical-gradient/) | 10 cells, k = 3 | `W = int x u dx` |
| [Shestakov](../shestakov-nonlinear/), `n_b = 0.05` | 10 cells, k = 3 | `N = int u dx` |
| [two-temperature](../two-temperature-critical-gradient/) | 10 cells, k = 3 | `W_i = int x T_i dx`, `W_e = int x T_e dx` |

The resolutions are the ones `../paper-figures/steady_modes.py` uses; three of
the four problems are cylindrical, and the stored energy there carries `V' = x`.
The two-temperature problem has one objective per channel, and the tolerance
given holds for both. Every run uses the benchmarks' own `Relative_tolerance =
1e-6` and `Absolute_tolerance = 1e-3`, with `EstimateObjectiveOnFinish` off so
that each column is the stopping test's own cost.

## What it measures

`|dG|/|G|` is the largest relative error over the objectives at the stop,
against `NewtonStep` at `SteadyStateStepTolerance = 1e-6` (a correction of
`1e-12` relative) at the same discretisation, so it is the algebraic error and
nothing else. The share is half the tolerance, the default budget. `res`, `jac`
and `solves` are residual evaluations, Jacobian builds and Jacobian solves; `adj`
is adjoint solves; visits are physics point-evaluations per collocation point.
Measured 2026-10-17:

| problem | mode | test | tol | share | \|dG\|/\|G\| | iters | res | jac | solves | adj | visits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Park | Newton | NewtonStep | | | 0 | 2 | 2 | 2 | 2 | 0 | 6 |
| Park | Newton | Objective | 1e-2 | 5e-3 | 0 | 2 | 2 | 2 | 2 | 0 | 6 |
| Park | Newton | Objective | 1e-4 | 5e-5 | 0 | 2 | 2 | 2 | 2 | 0 | 6 |
| Park | PseudoTransient | NewtonStep | | | 7.7e-16 | 3 | 3 | 3 | 3 | 0 | 8 |
| Park | PseudoTransient | Objective | 1e-2 to 1e-4 | | 7.7e-16 | 3 | 3 | 3 | 3 | 0 | 8 |
| Jardin | Newton | NewtonStep | | | 1.5e-8 | 5 | 5 | 5 | 5 | 0 | 12 |
| Jardin | Newton | Objective | 1e-2 | 5e-3 | 3.5e-5 | **4** | 4 | 4 | 4 | 0 | **10** |
| Jardin | Newton | Objective | 1e-3, 1e-4 | | 1.5e-8 | 5 | 5 | 5 | 5 | 0 | 12 |
| Jardin | PseudoTransient | NewtonStep | | | 4.2e-13 | 7 | 7 | 7 | 7 | 0 | 16 |
| Jardin | PseudoTransient | Objective | 1e-2, 1e-3 | | 1.9e-7 | **6** | 6 | 6 | 6 | 0 | **14** |
| Jardin | PseudoTransient | Objective | 1e-4 | 5e-5 | 4.2e-13 | 7 | 7 | 7 | 7 | 0 | 16 |
| Shestakov | Newton | NewtonStep | | | **6.5e-4** | 14 | 14 | 14 | 14 | 0 | 30 |
| Shestakov | Newton | Objective | 1e-2 | 5e-3 | 6.5e-4 | 14 | 14 | 14 | 14 | 0 | 30 |
| Shestakov | Newton | Objective | 1e-3, 1e-4 | | 6.4e-8 | 16 | 16 | 16 | 16 | 0 | 34 |
| Shestakov | PseudoTransient | NewtonStep | | | **1.8e-3** | 15 | 15 | 15 | 15 | 0 | 32 |
| Shestakov | PseudoTransient | Objective | 1e-2 | 5e-3 | 1.8e-3 | 15 | 15 | 15 | 15 | 0 | 32 |
| Shestakov | PseudoTransient | Objective | 1e-3 | 5e-4 | 3.0e-6 | 17 | 17 | 17 | 17 | 0 | 36 |
| Shestakov | PseudoTransient | Objective | 1e-4 | 5e-5 | 6.9e-10 | 18 | 18 | 18 | 18 | 0 | 38 |
| two-temp. | Newton | NewtonStep | | | 2.0e-8 | 5 | 5 | 5 | 5 | 0 | 12 |
| two-temp. | Newton | Objective | 1e-2 | 5e-3 | 1.3e-3 | **3** | 3 | 3 | 3 | 0 | **8** |
| two-temp. | Newton | Objective | 1e-3, 1e-4 | | 2.0e-8 | 5 | 5 | 5 | 5 | 0 | 12 |
| two-temp. | PseudoTransient | NewtonStep | | | 8.0e-15 | 6 | 6 | 6 | 6 | 0 | 14 |
| two-temp. | PseudoTransient | Objective | 1e-2 to 1e-4 | | 8.0e-15 | 6 | 6 | 6 | 6 | 0 | 14 |

**Every stop is inside its share**, on every problem, mode and tolerance — the
closest are Shestakov under `PseudoTransient` at `1e-2`, 1.8e-3 against 5e-3,
and the two-temperature problem under Newton at `1e-2`, 1.3e-3. Most are
far inside it, and that is by construction rather than luck: the estimate is
made at the iterate the last Newton correction starts from, and the correction
is then applied, so the state returned is one Newton step better than the state
tested. Under full Newton that step is quadratic, which is why a tolerance of
`1e-3` on Jardin returns an error of `1.5e-8`.

**What it saves is the last iteration or two, and only when the tolerance is
loose.** Jardin at `1e-2` takes 4 Newton iterations against `NewtonStep`'s 5;
the two-temperature problem takes 3 against 5, a third fewer visits. At `1e-3`
and tighter the two tests stop on the same iteration on Park, Jardin and the
two-temperature problem: Newton's quadratic convergence leaves no iteration in
which the objective has converged and the solution has not, at these
`Relative_tolerance`s. Under `PseudoTransient` most of the cost is the damped
continuation, which the objective test does not change, so the saving there is
at most the one undamped iteration.

**What it buys is the accuracy that was asked for, which `NewtonStep` does not
promise.** On Shestakov, `NewtonStep` at its default stops with the particle
inventory `6.5e-4` (Newton) and `1.8e-3` (`PseudoTransient`) away from the
fixed point. Nothing in `SteadyStateStepTolerance = 0.33` says that it would;
the degenerate flux makes this problem's corrections small long before its
objective is. Asked for `1e-3`, the objective test spends two more iterations
and returns `6.4e-8` and `3.0e-6`. That is the case for the test: the tolerance
is on the quantity the study reports, and the run says whether it was met.

**It asks the physics for nothing.** Every row's visits are its iterations' own
residuals and builds, plus the sweep that builds the initial `sigma` and the
derivative sweep of the end-of-run gradient `solveAdjoint` asks for, exactly as
for `NewtonStep`; and no row makes an adjoint solve. The wall-clock cost is the
objective's `dg` once per undamped iteration, which for the NumPy objectives here
is lost in the noise of a few milliseconds per run (`benchmark.py` prints it).
An objective whose `dg` is expensive — `AutodiffAdjointProblem` differentiates
by forward-mode autodiff at every node — pays that per iteration.

## Only the named objectives are controlled

The test stops when the objectives are converged and says nothing about anything
else the run reports. On the two-temperature problem at `1e-2` the stored
energies are within `1.3e-3` of the fixed point and the temperature profiles,
pointwise, within `2.7e-3` — twice as far. (On Jardin at `1e-2` the two agree,
`3.5e-5` each, because the iterate there is the linear answer with a slightly
wrong slope; that is a property of the problem, not of the test.) A quantity
the objective does not weigh — a gradient at the edge, a flux at mid-radius —
has whatever error it has. Name every output that matters as an objective: the
test holds all of them, each to its own tolerance if `ObjectiveTolerance` is
given one per objective.

## What the numbers do not say

The error measured here is the **algebraic** error: the stop against the discrete
fixed point. The error of that fixed point against the continuum is the other
half of the budget, and nothing estimates it yet — `ObjectiveBudget` reserves
its share, and the steady test spends only its own. The discretisation estimate
is the next phase of
`docs/superpowers/specs/2026-10-17-objective-error-control-design.md`.

`toy-model/` is not among the problems: it is linear diffusion with no source
under another name (see its README), and Park already covers a linear problem.
