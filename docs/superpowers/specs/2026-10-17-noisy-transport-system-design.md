# Error control on an objective, and the `NoisyTransportSystem` interface — design

**Status:** draft, 2026-10-17. Branch `feat/noisy-transport-system`.
**Builds on:** `docs/superpowers/notes/2026-10-17-noisy-flux-error-bars-notes.md`
(the measurements this rests on) and
`docs/superpowers/specs/2026-10-11-steady-state-termination-design.md`
("Noise: phase 2", which this replaces for noisy cases).

## Goal

A user asks for an **output** to an accuracy -- "the stored energy to within
1%" -- and MaNTA delivers it at the least cost. That is one feature with two
uses:

* **Deterministic systems** -- every existing case. The tolerance on a chosen
  objective decides when the solve stops and, optionally, how the
  discretisation adapts, in place of tolerances on the whole solution that
  carry no information about which output matters.
* **Noisy systems** -- an expensive flux model with an error bar, such as GX
  through QUENDS. MaNTA also works out the accuracy it needs from the flux model
  at each point and each stage of the solve and passes those bounds to it. The
  model meets them however it can: by running longer, or by answering from a
  surrogate whose predicted error is already small enough.

**Every piece of the adjoint machinery this needs, and what exists already, is
exposed to Python.**

## What exists

From a survey of the tree as of `7f20f36`:

* `estimateObjective()` (`SystemSolver.cpp`) computes `corrected = G - G_y .
  delta`, where `J delta = F`. That equals `G + lambda^T F`, the dual-weighted
  estimate of the **algebraic** error, and an `uncertainty` bound
  `||G_y|| ||delta||`. It runs at the end of each steady solve or slice; Python
  reads it through `objectiveEstimate()`. Nothing consumes it during a solve
  (`TODO`). Per objective it re-runs `ComputePhysicsDerivatives` through
  `initializeMatricesForAdjointSolve`, which is a physics call it does not need.
* **Nothing estimates the discretisation error of an objective.**
  `DegreeAdaptation` is driven by `||u* - u_h|| / ||u_h||` (`DegreeTolerance`),
  and `MeshAdaptation` by each cell's modal decay. Neither is weighted by an
  adjoint.
* **The adjoint state is not exposed.** `solveAdjointState` leaves it in
  `adjoint_squ` (per cell, `[sigma | q | u | aux]`), `adjoint_lambdas` and
  `adjoint_field`. Python receives `G`, `G_p` and `G_p_boundary` only.
* **Nothing computes `dG` with respect to the flux values at the physics
  nodes.** The pieces are in `accumulateAdjointGradients`, contracted with
  `dSigma/dp`.
* No tolerance on any output exists. Stored energy exists as an objective in
  `AdjointPlasma` and `python-physics/stellarator`.

## Architecture: loosely coupled (decided 2026-10-17)

```
 MaNTA core (C++)                        objective error estimates, tolerance,
   |                                     budget, stopping, adaptation, requests
   |  interface 1: NoisyTransportSystem  (zero: today's TransportSystem)
   v
 a flux system (Python, outside core)    DirectModel | FilterProxy | GPProxy | ...
   |
   |  interface 2: expensive model       (the surrogates only)
   v
 GX + QUENDS, or a test box
```

**The core holds everything that needs J, lambda or the discretisation**: the
objective and its tolerance, the three error estimates, the budget, stopping,
adaptation, and the accuracy requests. **It knows nothing about how a system
meets a request**, and a surrogate is one implementation of interface 1 among
several.

* **Interface 1 has implementations already, and they are the design check.**
  The toy model's per-point Kalman filter (`reporting.py`) and a direct GX
  wrapper with no surrogate must both fit it. If they do not, the interface is
  wrong, and that is found out before any GP exists.
* **The surrogate side changes faster than the core.** Kernels, inputs and
  sampling rules belong in Python and JAX or GPyTorch, outside the C++ core and
  its required CI legs, as `manta.jax` is kept optional.
* **The core can be verified without GX**, and much of it without noise at all.
  Did the solve deliver W to the requested 1%, on a case whose W is known in
  closed form? The toy model has one.

## The error in an objective

For an objective `G = int g dx` (`AdjointProblem`), the delivered value differs
from the true one in three parts:

| part | what it is | estimate | exists? |
|---|---|---|---|
| discretisation | `G(u_h*) - G(u*)`, at a converged discrete solution | `eta_disc`, dual-weighted, below | no |
| algebraic | the solve stopped before `F = 0` | `lambda^T F` | yes, `estimateObjective` |
| statistical | the physics outputs carry noise | `sqrt(w^T Sigma w)`, below | no |

The tolerance `eps_G` is split between them by `ObjectiveBudget`. For a
deterministic system the statistical share is zero.

### Discretisation: a dual-weighted estimate

The standard form, `eta_disc = R(u_h)(z - z_h)`, needs an adjoint `z` more
accurate than the discrete one. Its per-cell contributions `eta_K` are the
adaptation indicator. Two candidates, to be chosen by measurement:

* **Enriched adjoint.** Solve the adjoint at degree `k+1`. It is linear: one
  assembly and one solve at the new degree, against the forward Jacobian's
  cost, with the physics derivatives evaluated at the `k+2` nodes. That is a
  physics call, which matters for an expensive model.
* **Postprocessed adjoint.** `Postprocessing` already reconstructs
  `u* in P_{k+1}` from `(u_h, q_h)` per cell, at no physics cost. The same
  local reconstruction applied to the adjoint's `(u, q)` blocks gives
  `z* in P_{k+1}`, and `eta_disc ~ R_{k+1}(u_h)(z* - z_h)` evaluates the
  residual on the `k+2` star nodes `Superconvergent` already uses.

**Chosen by effectivity**, `eta_disc / (G(u_h*) - G(u*))`, on cases with exact
objectives: the toy model (`W = int u dx` in closed form), Park,
`MMSConvergenceTests`' manufactured problems, and Shestakov. An estimator near 1
across degrees and meshes is kept; one that is off by a consistent factor is
calibrated; one that is erratic is rejected. Note the trap `CLAUDE.md` records
for `diag(w)` against the mass matrix: two quadratures agreeing is not evidence
that either is right, so the reference is always a closed form.

### Statistical: the sensitivity to the physics outputs

Let `Gamma_m` be any physics output a system reports as uncertain -- the flux
`sigma_hat` or a source `S`, per channel -- at physics node `m`. The residual
reads it linearly: the sigma row is `A sigma_h + (I_h sigma_hat, phi)`, and the
sources enter the u row the same way, so `dF/dGamma_m` is one column of the
cell's interpolation operator (`InterpolateOntoBasis`, or `A9` under
`Superconvergent`). With `J^T z = dG/dy`,

```
w_m = dG/dGamma_m = -z^T dF/dGamma_m
Var(G) = sum_m w_m^T Sigma_m w_m
```

`Sigma_m` is the channel covariance at node `m`. Nodes from a flux-tube code are
independent; channels at one node are not, since heat and particle flux come
from one run. **A sign or operator error in `w` is invisible to every
aggregate** -- the `diag(w)` trap again -- so `w` is pinned per node by finite
differences of `G` under a single-node perturbation of a case's flux. `w` is
also useful for deterministic cases: it says where the objective is sensitive
to the physics, which is a model-uncertainty statement in its own right.

### No physics calls for the adjoint

Each estimate above needs an adjoint solve at the current state, possibly once
per Newton iteration. **None may evaluate the physics.** The adjoint operator is
the transpose of the forward Jacobian MaNTA has just built, so it is solved
with that build's factorised blocks, transposed. `estimateObjective` and
`initializeMatricesForAdjointSolve` re-evaluate `ComputePhysicsDerivatives`
today, and are to be refactored to reuse the build. Against GX that difference
is the whole cost. The enriched discretisation estimator is the one exception,
and that is a reason to prefer the postprocessed one if its effectivity holds.
`w` only needs to be roughly right: an error of 20% moves the request
allocation little and the variance estimate by 20%, which a safety factor
covers.

## Objective tolerance on a deterministic system

Configuration keys, following the `UpperCamelCase` convention:

* `ObjectiveTolerance` -- `eps_G`, relative to `|G|`, per objective.
  `ObjectiveAbsoluteTolerance` is the floor for an objective near zero. With
  `ng > 1` each objective carries its own, and every test below must hold for
  all of them.
* `ObjectiveBudget` -- the shares (discretisation, algebraic, statistical).
  The default is half and half for a deterministic system.
* `AdaptationTarget = "Solution" | "Objective"` -- what `DegreeAdaptation` and
  `MeshAdaptation` adapt to. `"Solution"` is today's behaviour and the default.

**Stopping.** `SteadyStateTest = "Objective"` stops the steady solve when
`|lambda^T F|` is below the algebraic share for every objective. It needs an
adjoint solve per Newton iteration of the undamped solve, from the factorised
Jacobian at no physics cost, and none during damped continuation, where the
state is far off and the estimate means little. **It controls only the named
objectives.** Anything else the run outputs may be less converged than the
NewtonStep test would leave it, and the docs must say so: name every output that
matters as an objective. The adjoint gradients `G_p` are taken at that state too.

**Adaptation.** Under `AdaptationTarget = "Objective"`:

* `DegreeAdaptation` replaces its control quantity `||u* - u_h|| / ||u_h||` with
  `|eta_disc| / (eps_disc |G|)`, in the same rule `dk = ceil(log_base(E / eps))`.
  The base needs remeasuring: functionals typically converge at a higher rate in
  `k` than the solution does.
* `MeshAdaptation` decides which end to grade from the distribution of
  `|eta_K|`, instead of each cell's modal decay.
* The plan machinery (`ParallelFill`, `MaxPhysicsBatch`) is unchanged.

## Exposing the adjoint to Python

The `Runner` (and `FFIRunner`, for the host-side reads) gains:

| call | returns | phase |
|---|---|---|
| `solve_adjoint(g=None)` | solves at the current state, on demand -- between slices, not only at the end -- reusing the last build | 1 |
| `adjoint_state(g)` | `z` as nodal arrays per variable: `sigma`, `q`, `u`, `aux`, the traces, `mu`, `psi` | 1 |
| `objective_gradient(g)` | `G_y`, the right-hand side, in the same layout | 1 |
| `objective_sensitivity(g)` | `w` for every physics output, per node and channel, with node positions | 1 |
| `objectiveEstimate()` | exists; gains the discretisation and statistical estimates and the per-cell `eta_K` | 2, 4 |
| `getAdjointGradients()` | exists | -- |

`WriteAdjoints()`, commented out at `Solver.cpp:350`, comes back as an option so
a run can write `z` beside `u`.

## Interface 1: `NoisyTransportSystem`

A `TransportSystem` whose physics outputs carry uncertainty, and which accepts
accuracy requests. Per evaluation call:

**MaNTA to the system**
* the points and states, as today; the point sets are known in advance through
  `EvaluationPlan`;
* the request: per-node, per-channel standard-error targets `s_m`, and the
  functional itself -- the weights `w` and `eps_noise`. A direct model wants
  the targets, since it chooses a run length per tube. A surrogate wants the
  functional, since its sampling rule minimises `Var(w^T Gamma)` over where to
  run;
* whether this is a value call or a derivative call;
* access to the current solution at arbitrary `x`, so a surrogate can run the
  model at points of its own choosing.

**The system to MaNTA**
* means, as today;
* the covariance `Sigma_m` per node, across channels; one standard error per
  channel at least;
* for derivative calls, the derivatives;
* per point: whether the target was met, and whether the value is flagged
  (QUENDS's no-steady-state fallback);
* the cost of the call, in the model's unit, for accounting and for `c_m`.

A system that ignores requests and reports no covariance is today's
`TransportSystem`. That is the zero of the interface, and every existing case
stays untouched.

### Accuracy requests

With the statistical share `eps_noise`, and a call at node `m` costing
`c_m / s_m^2` -- a time average, `s ~ T^(-1/2)` -- the cheapest targets meeting
`sum w_m^2 s_m^2 <= eps_noise^2` are

```
s_m^2 = eps_noise^2 sqrt(c_m) / (|w_m| sum_j sqrt(c_j) |w_j|)
total cost = (sum_j sqrt(c_j) |w_j|)^2 / eps_noise^2
```

Nodes the objective is sensitive to get tight targets, and the rest loose ones.
With several objectives, a node gets the tightest target any of them needs.
**Requests are loose early and tighten as the solve converges**, like the
forcing term in inexact Newton: the share at a call is
`max(eps_noise, eta |lambda^T F|)`. Early calls are cheap -- short runs, or the
surrogate's prediction.

### Stopping under noise

The algebraic and statistical tests both have to hold: `|lambda^T F|` below its
share and `sqrt(w^T Sigma w)` below `eps_noise`. `lambda^T F` is itself noisy,
with the statistical error as its floor, so its share must not be set below
that. The residual and the Newton correction measure neither part, as the
filtered solve showed and as Parker et al. found for the residual. A target the
system reports it cannot meet ends the solve with an outcome that says so, not a
hang.

## Interface 2: the expensive model (surrogate side, not the core)

`run(inputs, budgets) -> mean, standard error (or covariance across channels),
flags, cost`, batched, since GX runs tubes in parallel. The inputs are what the
model takes, such as `(rho, n, T, gradients)`, and the budgets a run length or a
target error. It lives with the surrogates.

## Phases

1. **The adjoint, exposed and free of physics calls.** `solve_adjoint`,
   `adjoint_state`, `objective_gradient` and `objective_sensitivity`, solved
   from the last forward build. `estimateObjective` is refactored onto the
   same path. `w` is verified per node by finite differences on the toy model,
   with `W = int u dx`. Every existing adjoint test is bit-for-bit unchanged.
2. **Objective tolerance, algebraic.** `ObjectiveTolerance`, `ObjectiveBudget`,
   `SteadyStateTest = "Objective"`. Measured on the deterministic benchmarks
   (toy, Park, Jardin, Shestakov, two-temperature): does it stop with `|G|`
   inside its share, and at what cost against `NewtonStep`?
3. **The discretisation estimate.** Both candidates are implemented and
   measured for effectivity, and one is kept. It is exposed through
   `objectiveEstimate()` with the per-cell `eta_K`.
4. **Adaptation keyed off the objective.** `AdaptationTarget = "Objective"` in
   `DegreeAdaptation` and `MeshAdaptation`. Measured: does the adapted run meet
   `eps_G` against the closed form, and at what cost against
   `AdaptationTarget = "Solution"` tuned to the same accuracy in `G`?
5. **Interface 1, reporting only.** Covariance and status come back through it,
   and MaNTA reports `G +- sqrt(w^T Sigma w)` but makes no requests yet.
   Implementations: `DirectModel` and `FilterProxy` on the toy box. The
   ensemble over seeds checks that the reported variance is calibrated.
6. **Error control under noise.** Requests with the forcing schedule, the
   three-part stopping test and its outcomes. The toy box gains a run-length
   knob, `s ~ n^(-1/2)`, with cost counted in effective samples. Measured: the
   coverage over seeds, and the cost against a solve at fixed accuracy.
7. **GPProxy**, outside the core, over interface 2. It answers from its
   posterior where `s_post <= s_m`, and otherwise samples the model where that
   most reduces `Var(w^T Gamma)`. Measured against `FilterProxy` at equal model
   cost, on the toy, on Shestakov with Parker et al.'s noise, and on the T3D
   ReLU benchmark.

Phases 1-4 are wanted for deterministic cases in their own right, and are the
candidates to reach `main` independently of this branch's `gp-toy-model`
ancestry.

## Open questions

* **Where the hooks live.** Virtuals on `TransportSystem` with null defaults,
  or a `NoisyTransportSystem` subclass in C++ bound to Python? The former keeps
  one physics interface. The latter keeps the uncertainty optional in type as
  well as in behaviour.
* **Time-dependent runs.** This design is for steady solves. In a transient the
  error in G accumulates over steps, and IDA's local error control is where it
  would join. An adjoint for a transient objective is a backward-in-time solve
  MaNTA does not have. Out of scope for now.
* **Uncertainty in the derivatives.** Newton needs J only roughly. The
  significance guard used the derivatives' errors, and a surrogate can supply
  them. Is that part of interface 1, or the system's own business?
* **The adjoint's own accuracy.** Each estimate uses `z` at the current state,
  whose error is first order in the state's error. Stopping can tolerate that.
  Whether adaptation can is measured in phase 4.
* **Field models and scalars.** `w` and `eta_disc` cover the transport physics
  first. The field notes record that the coupled adjoint lacks a
  `dg/dgeometry` hook, so a `G` that reads geometry gets `G_field = 0`.
