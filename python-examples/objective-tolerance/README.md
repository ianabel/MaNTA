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

The error measured above is the **algebraic** error: the stop against the
discrete fixed point. The error of that fixed point against the continuum is the
other half of the budget: `EstimateDiscretisationError` estimates it
(`docs/adjoints.rst`), and `AdaptationTarget = "Objective"` adapts the degree and
the mesh to it, below.

`toy-model/` is not among the problems: it is linear diffusion with no source
under another name (see its README), and Park already covers a linear problem.

# Adapting to the objective

`AdaptationTarget = "Objective"` drives `DegreeAdaptation` and `MeshAdaptation`
from the objectives rather than the solution: the degree rises until every
objective's estimated discretisation error is within the discretisation share of
`ObjectiveTolerance`, and the mesh is graded -- where the modal decay says, as
for the solution target -- only if the sample does not meet that already.
`docs/adaptivity.rst` ("Adapting to an objective") has the rule, the
measurements behind it, and why the per-cell estimate does not choose the end.

    python adaptation.py            # the table below, seconds
    python adaptation.py --sweep    # whether eps_G is met, over 210 runs
    python adaptation.py --starts   # ... and from 35 non-uniform meshes

The problems all have a closed-form objective, so the error is against the
continuum, not the estimate: a manufactured nonlinear problem,
`-d/dx[(1 + u^2) u'] = S` with `u = sin(pi x)` and `G = int u^2 dx = 1/2`;
Park's stored energy `int x u dx = (e - 2)/2`; `u = x - x^(4/3)`, Shestakov's
axis singularity in a linear problem, with `int u dx = 1/14` (and in the sweeps
its mirror image, singular at `x = 1`); and Shestakov's
inventory `int n dx` at `n_b = 0.05`, from his closed-form steady state.

Each objective-target run is set against the solution target at the default
`DegreeTolerance = 1e-6` and at a tuned one, the loosest half-decade at which
the error in `G` is within `eps_G` — which needs the closed form to find. Both
targets stop each solve with the same `Objective` test, so only what a level is
measured by differs. Columns: the relative error in `G` from `Runner.G()`; the
degrees solved and the final mesh; whole sweeps of the physics (flux and
derivative) / point evaluations. A graded mesh is 10 cells with the wall cell
at 6.6e-6 of the domain. Measured 2026-10-18:

| problem, driver, start | eps_G | Objective | Solution, 1e-6 | Solution, tuned |
|---|---|---|---|---|
| manufactured, degree, k = 1, 4 cells | 1e-3 | 1.8e-5; k 1, 2; 25 / 368 | 2.5e-12; k 1, 4, 7; 24 / 504 | 1e-1: 1.8e-5; k 1, 2; 18 / 240 |
| | 1e-5 | 2.2e-8; k 1, 4; 27 / 488 | 2.5e-12; k 1, 4, 7; 26 / 552 | 3e-2: 8.3e-7; k 1-3; 26 / 392 |
| Park, degree, k = 1, 4 cells | 1e-3 | 1.9e-5; k 1, 3; 18 / 320 | 2.2e-8; k 1, 4, 5; 18 / 384 | 1e-2: 9.4e-5; k 1, 2; 12 / 168 |
| | 1e-5 | 8.1e-8; k 1, 4; 18 / 356 | 2.2e-8; k 1, 4, 5; 18 / 384 | 1e-4: 8.1e-8; k 1, 4; 12 / 216 |
| Park, mesh, k = 3, 10 cells | both | 4.6e-7; k 3, uniform; 9 / 490 | 3.6e-10; k 3, 4, uniform; 12 / 660 | 1e-1: 4.6e-7; k 3, uniform; 6 / 300 |
| x - x^(4/3), mesh, k = 3, 10 cells | 1e-3 | 5.8e-6; k 3, **uniform**; 9 / 490 | 2.7e-7; k 3, 6, 9, 12, graded; 30 / 2580 | 3e-2: 3.5e-4; k 3, 4, graded; 18 / 960 |
| | 1e-5 | 3.4e-6; k 3, 6-9, graded; 55 / 4610 | 2.7e-7; k 3, 6, 9, 12, graded; 30 / 2580 | 1e-3: 7.9e-6; k 3, 5-8, graded; 36 / 2640 |
| Shestakov, mesh, k = 3, 10 cells | 1e-3 | 4.9e-5; k 3, graded; 52 / 2680 | 2.8e-7; k 3, 4, graded; 50 / 2560 | 1e-1: 4.9e-5; k 3, graded; 44 / 2200 |
| | 1e-5 | 5.0e-7; k 3, graded; 62 / 3180 | 7.3e-7; k 3, 4, graded; 60 / 3060 | 1e-1: 5.0e-7; k 3, graded; 54 / 2700 |
| Shestakov, degree only, k = 3, 10 cells | both | 1.3e-3, at the ceiling | 1.3e-3, at the ceiling | not reachable |

**It meets `eps_G` wherever that can be met, without knowing the answer**, and
where it cannot — Shestakov's problem by the degree alone, whose error the
singularity caps — it stops at `MaxPolynomialDegree` with a warning, as the
solution target does. The tuned solution target matches it, but its
`DegreeTolerance` runs from 1e-1 to 1e-4 across the rows, and no single value
serves: 1e-1, enough for four rows, leaves Park's stored energy 12x outside 1e-3.

**It pays for the estimate**: per level, the objective estimate's residual and
Jacobian build and the discretisation estimate's two sweeps at the enriched
nodes. Where both targets take the same degrees that is 1.2-1.7x the tuned
target's point evaluations. Against the default `DegreeTolerance` it spends
fewer in seven runs of ten, and 5x fewer on `x - x^(4/3)` at 1e-3, where the
uniform sample already meets the tolerance and it grades nothing, while the
solution target grades and climbs to k = 12. On the same problem at 1e-5 it
climbs one degree at a time, the estimate falling only 2.5x per degree on the
graded mesh against the 10x the rule assumes, and takes a level more than the
default.

**From uniform meshes** (`--sweep`, 210 runs: the manufactured problem and Park
by both drivers, `x - x^(4/3)`, its mirror image and Shestakov's by
`MeshAdaptation`, from k = 3 and 4 on 5, 10 and 20 cells, at `eps_G` from 1e-3 to
1e-7): 176 converged, every one within `eps_G`, the closest at 0.81 of it, and
none outside it; 34 stopped at the ceiling with a warning, 4 of them within
`eps_G` anyway. All 34 are meshes that cannot carry the accuracy, and the
solution target settles on the same mesh with the same error: Shestakov's at 5
cells grades a layer holding the source edge at `x = 0.1` inside a cell, and at
20 cells puts 19 cells into `[0, 0.05]` and one across the edge; `x - x^(4/3)`
and its mirror at 20 cells past 1e-6 and 10 cells past 1e-7 run the single bulk
cell out of degree.

**From non-uniform meshes** (`--starts`, 350 runs: `MeshAdaptation` from k = 3 on
35 meshes -- graded at either end, graded nine cells deep, and random -- for five
of those problems at 1e-5 and 1e-7): 232 converged within `eps_G` and one outside
it, the mirrored singularity from a random six-cell mesh at 1e-7, where the
estimate in the split wall cell has the wrong sign and is 30x short (5.9e-7
against 1e-7). 117 stopped at the ceiling, all on the singular problems --
Shestakov's source edge inside a cell, an end the modal decay cannot judge, or a
wall cell the degree no longer helps. Every smooth run converged within its
tolerance on its starting mesh. The solution target at the default
`DegreeTolerance` over the same starts converged outside `eps_G` 21 times, all on
Shestakov's problem and by up to 1.2e-2, never having been told it.

The estimate underestimates an error that decays only algebraically in `k`, so on
the singular problems the margin is the budget's other half rather than the
estimate's.
