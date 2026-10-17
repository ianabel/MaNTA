# Error control on an objective — design

**Status:** agreed 2026-10-17; phases 1-4 below. Branch
`feat/objective-error-control`, from `main`.

## Goal

A user asks for an **output** to an accuracy -- "the stored energy to within
1%" -- and MaNTA delivers it at the least cost. The tolerance on a chosen
objective decides when the steady solve stops and, optionally, how the
discretisation adapts. Today those are decided by tolerances on the whole
solution, which carry no information about which output matters.

**Every piece of adjoint machinery involved, and what exists already, is exposed
to Python.**

This is the deterministic half of a larger design. The other half, on branch
`feat/noisy-transport-system`, adds a third, statistical part to the error
budget and passes accuracy requests to a flux model that reports an error bar.
Nothing here depends on it. The pieces it will need -- the sensitivity of G to
the physics outputs, and adjoint solves that make no physics call -- are built
here because they are useful without noise.

## What exists

As of `87331fb`:

* `estimateObjective()` (`SystemSolver.cpp`) computes `corrected = G - G_y .
  delta`, where `J delta = F`. That equals `G + lambda^T F`, the dual-weighted
  estimate of the **algebraic** error, and an `uncertainty` bound
  `||G_y|| ||delta||`. It runs at the end of each steady solve or slice, and
  Python reads it through `objectiveEstimate()`. Nothing consumes it during a
  solve (`TODO`). Per objective it re-runs `ComputePhysicsDerivatives` through
  `initializeMatricesForAdjointSolve`, which is a physics call it does not need.
* **Nothing estimates the discretisation error of an objective.**
  `DegreeAdaptation` is driven by `||u* - u_h|| / ||u_h||` (`DegreeTolerance`),
  and `MeshAdaptation` by each cell's modal decay. Neither is weighted by an
  adjoint.
* **The adjoint state is not exposed.** `solveAdjointState` leaves it in
  `adjoint_squ` (per cell, `[sigma | q | u | aux]`), `adjoint_lambdas` and
  `adjoint_field`. Python receives `G`, `G_p` and `G_p_boundary` only.
* **Nothing computes `dG` with respect to the physics outputs at the nodes.**
  The pieces are in `accumulateAdjointGradients`, contracted with `dSigma/dp`.
* No tolerance on any output exists. Stored energy exists as an objective in
  `AdjointPlasma` and `python-physics/stellarator`.

## The error in an objective

For an objective `G = int g dx` (`AdjointProblem`), the delivered value differs
from the true one in two parts:

| part | what it is | estimate |
|---|---|---|
| discretisation | `G(u_h*) - G(u*)`, at a converged discrete solution | `eta_disc`, dual-weighted, below (new) |
| algebraic | the solve stopped before `F = 0` | `lambda^T F` (exists, in `estimateObjective`) |

The tolerance `eps_G` is split between them by `ObjectiveBudget`.

### Discretisation: a dual-weighted estimate

The standard form, `eta_disc = R(u_h)(z - z_h)`, needs an adjoint `z` more
accurate than the discrete one. Its per-cell contributions `eta_K` are the
adaptation indicator. Two candidates, to be chosen by measurement:

* **Enriched adjoint.** Solve the adjoint at degree `k+1`. It is linear: one
  assembly and one solve at the new degree, with the physics derivatives
  evaluated at the `k+2` nodes, which is a physics call.
* **Postprocessed adjoint.** `Postprocessing` already reconstructs
  `u* in P_{k+1}` from `(u_h, q_h)` per cell, at no physics cost. The same local
  reconstruction applied to the adjoint's `(u, q)` blocks gives
  `z* in P_{k+1}`, and `eta_disc ~ R_{k+1}(u_h)(z* - z_h)` evaluates the
  residual on the `k+2` star nodes `Superconvergent` already uses.

**Chosen by effectivity**, `eta_disc / (G(u_h*) - G(u*))`, on cases with exact
objectives: `MMSConvergenceTests`' manufactured problems, Park, and Shestakov,
and a closed-form `G` such as `int u dx` wherever the exact solution is known.
An estimator near 1 across degrees and meshes is kept; one off by a consistent
factor is calibrated; one that is erratic is rejected. The reference is always a
closed form: two quadratures agreeing is not evidence that either is right (the
`diag(w)` trap in `CLAUDE.md`).

### The sensitivity to the physics outputs

Let `Gamma_m` be a physics output -- the flux `sigma_hat` or a source `S`, per
channel -- at physics node `m`. The residual reads it linearly: the sigma row is
`A sigma_h + (I_h sigma_hat, phi)`, and the sources enter the u row the same
way, so `dF/dGamma_m` is one column of the cell's interpolation operator
(`InterpolateOntoBasis`, or `A9` under `Superconvergent`). With `J^T z = dG/dy`,

```
w_m = dG/dGamma_m = -z^T dF/dGamma_m
```

This says where the objective is sensitive to the physics, which is a statement
about model uncertainty in its own right; the noisy design builds on it. **A sign
or operator error in `w` is invisible to every aggregate**, so it is pinned per
node by finite differences of `G` under a single-node perturbation of a case's
flux and of its source.

### No physics calls for the adjoint

Each estimate needs an adjoint solve at the current state, possibly once per
Newton iteration. **None may evaluate the physics.** The adjoint operator is the
transpose of the forward Jacobian MaNTA has just built, so it is solved with that
build's blocks, transposed. `estimateObjective` and
`initializeMatricesForAdjointSolve` re-evaluate `ComputePhysicsDerivatives`
today, and are refactored to reuse the build. For an expensive flux model that
difference is the whole cost. The enriched discretisation estimator is the one
exception, and that is a reason to prefer the postprocessed one if its
effectivity holds.

**This must not move any existing answer.** The adjoint gradients, the objective
estimate and every adjoint test are bit-for-bit unchanged by the refactor, or
the difference is explained and measured. The forward/adjoint lockstep rule in
`CLAUDE.md` ("Adjoints") applies: the transposed blocks must be the ones
`initializeMatricesForAdjointSolve` builds today, block for block.

## Configuration

Following the `UpperCamelCase` convention, all in `ConfigSchema.cpp`:

* `ObjectiveTolerance` -- `eps_G`, relative to `|G|`, per objective.
  `ObjectiveAbsoluteTolerance` is the floor for an objective near zero. With
  `ng > 1` each objective carries its own, and every test below must hold for
  all of them.
* `ObjectiveBudget` -- the shares (discretisation, algebraic). Default half and
  half.
* `SteadyStateTest = "Objective"` -- see Stopping.
* `AdaptationTarget = "Solution" | "Objective"` -- what `DegreeAdaptation` and
  `MeshAdaptation` adapt to. `"Solution"` is today's behaviour and the default.

## Stopping

`SteadyStateTest = "Objective"` stops the steady solve when `|lambda^T F|` is
below the algebraic share for every objective. It needs an adjoint solve per
Newton iteration of the undamped solve, from the factorised Jacobian at no
physics cost, and none during damped continuation, where the state is far off
and the estimate means little. **It controls only the named objectives.**
Anything else the run outputs may be less converged than the `NewtonStep` test
would leave it, and the docs must say so: name every output that matters as an
objective. The adjoint gradients `G_p` are taken at that state too.

## Adaptation

Under `AdaptationTarget = "Objective"`:

* `DegreeAdaptation` replaces its control quantity `||u* - u_h|| / ||u_h||` with
  `|eta_disc| / (eps_disc |G|)`, in the same rule `dk = ceil(log_base(E / eps))`.
  The base is remeasured: functionals typically converge at a higher rate in
  `k` than the solution does.
* `MeshAdaptation` decides which end to grade from the distribution of
  `|eta_K|`, instead of each cell's modal decay.
* The plan machinery (`ParallelFill`, `MaxPhysicsBatch`) is unchanged.

## Exposing the adjoint to Python

The `Runner`, and `FFIRunner` for the host-side reads, gains:

| call | returns | phase |
|---|---|---|
| `solve_adjoint(g=None)` | solves at the current state, on demand -- between slices, not only at the end -- reusing the last build | 1 |
| `adjoint_state(g)` | `z` as nodal arrays per variable: `sigma`, `q`, `u`, `aux`, the traces, the scalars, the field DOFs | 1 |
| `objective_gradient(g)` | `G_y`, the right-hand side, in the same layout | 1 |
| `objective_sensitivity(g)` | `w` for the flux and the sources, per node and channel, with node positions | 1 |
| `objectiveEstimate()` | exists; gains the discretisation estimate and the per-cell `eta_K` | 3 |
| `getAdjointGradients()` | exists | -- |

`WriteAdjoints()`, commented out at `Solver.cpp:350`, comes back as an option so
a run can write `z` beside `u`. The committed stub `_manta.pyi` is regenerated
(`stubs-update`), and `__init__.pyi` and `docs/adjoints.rst` describe the new
calls.

## Phases

1. **The adjoint, exposed and free of physics calls.** The four calls above,
   solved from the last forward build; `estimateObjective` refactored onto the
   same path. `w` verified per node by finite differences. Every existing
   adjoint test unchanged.
2. **Objective tolerance, algebraic.** `ObjectiveTolerance`,
   `ObjectiveAbsoluteTolerance`, `ObjectiveBudget`,
   `SteadyStateTest = "Objective"`. Measured on the deterministic benchmarks
   under `python-examples/` (Park, Jardin, Shestakov, two-temperature): does it
   stop with the error in G inside its share, against a tightly converged
   reference, and at what cost against `NewtonStep`?
3. **The discretisation estimate.** Both candidates implemented and measured
   for effectivity; one kept, exposed through `objectiveEstimate()` with the
   per-cell `eta_K`.
4. **Adaptation keyed off the objective.** `AdaptationTarget = "Objective"` in
   `DegreeAdaptation` and `MeshAdaptation`. Measured: does the adapted run meet
   `eps_G` against a closed form or a converged reference, and at what cost
   against `AdaptationTarget = "Solution"` tuned to the same accuracy in G?

## Open questions

* **The adjoint's own accuracy.** Each estimate uses `z` at the current state,
  whose error is first order in the state's error. Stopping can tolerate that.
  Whether adaptation can is measured in phase 4.
* **Time-dependent runs** are out of scope. An adjoint for a transient
  objective is a backward-in-time solve MaNTA does not have.
* **Field models and scalars.** `w` and `eta_disc` cover the transport physics
  first. The field notes record that the coupled adjoint lacks a `dg/dgeometry`
  hook, so a `G` that reads geometry gets `G_field = 0`.
