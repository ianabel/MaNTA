# Error-controlled MaNTA and the `NoisyTransportSystem` interface — design

**Status:** draft, 2026-10-17. Branch `feat/noisy-transport-system`.
**Builds on:** `docs/superpowers/notes/2026-10-17-noisy-flux-error-bars-notes.md`
(the measurements this rests on) and
`docs/superpowers/specs/2026-10-11-steady-state-termination-design.md`
("Noise: phase 2", which this replaces for noisy cases).

## Goal

A user asks for an **output** to an accuracy -- "the stored energy to within
1%" -- and MaNTA delivers it at the least cost in calls to an expensive, noisy
flux model such as GX through QUENDS. MaNTA works out what accuracy it needs
from the flux model at each point and at each stage of the solve, and passes
those bounds to the model. The model meets them however it can: by running
longer, or by answering from a surrogate whose predicted error is already small
enough.

## Architecture: loosely coupled (decided 2026-10-17)

Three layers, joined by two interfaces:

```
 MaNTA core (C++)                       error control: objective tolerance,
   |                                    adjoint weights, budget, stopping
   |  interface 1: NoisyTransportSystem
   v
 a flux system (Python, outside core)   DirectModel | FilterProxy | GPProxy | ...
   |
   |  interface 2: expensive model      (GPProxy and FilterProxy only)
   v
 GX + QUENDS, or a test box
```

**The core holds everything that needs J, lambda or the discretisation:** the
objective and its tolerance, the adjoint weights, the error budget, the
per-call accuracy requests, and the stopping test. **It knows nothing about how
a system meets a request.** A surrogate is one implementation of interface 1
among several.

Why this split:

* **Interface 1 has implementations already**, and they are the design check.
  The toy model's per-point Kalman filter (`reporting.py`) and a direct GX
  wrapper with no surrogate must both fit it. If they do not, the interface is
  wrong, and that can be found out before any GP exists.
* **The surrogate side changes faster than the core.** Kernels, inputs and
  sampling rules belong in Python and JAX or GPyTorch, outside the C++ core and
  its required CI legs, as `manta.jax` is kept optional.
* **The core can be verified without GX.** Did the solve deliver W to the
  requested 1%, over seeds, on a box whose truth is known? That measures the
  core and the error control apart from the quality of any surrogate.
* **Tight coupling would buy little.** What a surrogate wants from the solver
  -- which points matter, how accurate this call must be, the current state --
  crosses the interface as data.

## The error in an objective

For an objective `G = int g dx` (`AdjointProblem`; stored energy exists in
`AdjointPlasma` and `python-physics/stellarator`), the delivered value differs
from the true one by three errors:

| part | what it is | how it is estimated |
|---|---|---|
| discretisation | `G(u_h*) - G(u*)`, at a converged discrete solution | not estimated anywhere today; phase 5 |
| algebraic | the solve stopped before `F = 0` | `lambda^T F`, which `estimateObjective` already computes as `corrected - value` |
| statistical | the flux values carry noise | `sqrt(w^T Sigma w)`, below; new |

The user's tolerance `eps_G` is split between them by an `ObjectiveBudget`.
Until phase 5 the discretisation share is the user's assertion, or is checked
by one refinement.

### Sensitivity of G to the flux values

Let `Gamma_m` be any physics output a system reports as uncertain -- the flux
`sigma_hat` or a source `S`, per channel -- at physics node `m`. The residual
reads it linearly: the sigma row is `A sigma_h + (I_h sigma_hat, phi)`, and the
sources enter the u row the same way, so `dF/dGamma_m` is one column of the
cell's interpolation operator (`InterpolateOntoBasis`, or `A9` under
`Superconvergent`). With `J^T z = dG/dy` the adjoint MaNTA already solves,

```
dG/dGamma_m = -z^T dF/dGamma_m = w_m
```

`w` comes from the adjoint solve `computeAdjointGradients` already makes, by
contracting the cell adjoint `adjoint_squ` with unit nodal perturbations instead
of `dSigma/dp`. Nothing computes it today, and lambda is not exposed to Python.
**A sign or operator error here is invisible to every aggregate**, as the
`diag(w)`-against-mass-matrix trap in `CLAUDE.md` was, so it is pinned per node
by finite differences of `G` under a single-node perturbation of a case's flux.

The statistical error is then `Var(G) = sum_m w_m^T Sigma_m w_m`, with
`Sigma_m` the channel covariance at node `m`. At a flux-tube code's resolution
nodes are independent and channels at one node are not: heat and particle flux
come from one run.

**The adjoint for `w` must not cost flux-model calls.** `estimateObjective`
re-runs `ComputePhysicsDerivatives` per objective through
`initializeMatricesForAdjointSolve`. Against GX that is an expensive call for a
quantity already in hand. The weights must be computed from the last forward
Jacobian build's blocks, transposed, with no physics evaluation.
`w` only needs to be roughly right. An error of 20% in it moves the allocation
below a little and the variance estimate by 20%, which a safety factor covers.

### Accuracy requests

Given `eps_noise`, the statistical share, MaNTA chooses a standard-error target
`s_m` per node and channel. Minimising cost, with a call at node `m` costing
`c_m / s_m^2` (a time average: `s ~ T^(-1/2)`), subject to
`sum w_m^2 s_m^2 <= eps_noise^2`:

```
s_m^2 = eps_noise^2 sqrt(c_m) / (|w_m| sum_j sqrt(c_j) |w_j|)
total cost = (sum_j sqrt(c_j) |w_j|)^2 / eps_noise^2
```

With equal costs, `s_m^2 = eps_noise^2 / (|w_m| sum |w_j|)`. Points W is
sensitive to get tight targets, and the rest get loose ones.

**Requests are loose early and tighten as the solve converges.** A far-off
iterate needs little accuracy, as the forcing term does in inexact Newton. The
request at a call is `max(eps_noise, eta * |current algebraic error in G|)`,
allocated as above. Early calls are then cheap: short GX runs, or the
surrogate's prediction.

### Stopping

Stop when the algebraic estimate `|lambda^T F|` is below its share **and** the
reported `sqrt(w^T Sigma w)` is below `eps_noise`. Under noise `lambda^T F` has
the statistical error as its own floor, so its share must not be set below it.
This is the stopping test the noise notes call for: the residual and the Newton
correction measure neither part, as the filtered solve showed and as Parker et
al. found for the residual. A target the system reports it cannot meet ends the
solve with an outcome that says so, not a hang.

## Interface 1: `NoisyTransportSystem`

A `TransportSystem` whose physics outputs carry uncertainty, and which accepts
accuracy requests. Per evaluation call:

**MaNTA to the system**
* the points and states, as today; the point sets are known in advance through
  `EvaluationPlan`;
* the request: either per-node, per-channel standard-error targets `s_m`, or
  the functional itself -- the weights `w` and `eps_noise`. A direct model wants
  the targets, since it chooses a run length per tube; a surrogate wants the
  functional, since its sampling rule minimises `Var(w^T Gamma)` over where to
  run. Both are sent, and the targets are computed from the functional;
* whether this is a value call or a derivative call;
* access to the current solution at arbitrary `x`, so a surrogate can run the
  model at points of its own choosing.

**The system to MaNTA**
* means, as today;
* the covariance `Sigma_m` per node, channel by channel: one standard error per
  channel at least;
* for derivative calls, the derivatives; their uncertainty is optional;
* per point: whether the target was met, and whether the value is flagged
  (QUENDS's no-steady-state fallback);
* the cost of the call, in the model's unit (tube-seconds, calls), for
  accounting and for `c_m`.

A system that ignores requests and reports no covariance is today's
`TransportSystem`. That is the zero of the interface, and every existing case
stays untouched.

## Interface 2: the expensive model (surrogate side, not the core)

`run(inputs, budgets) -> mean, standard error (or covariance across channels),
flags, cost`, batched, because GX runs tubes in parallel. The inputs are what
the model takes, such as `(rho, n, T, gradients)`, and the budgets are a run
length or a target error. It lives with the surrogates, not in MaNTA.

## Phases

1. **Weights.** `w = dG/dGamma` per node and channel, from the forward
   Jacobian's blocks with no physics call, exposed to Python. Verified per node
   by finite differences on the toy model, with `W = int u dx`. Then
   `Var(G) = w^T Sigma w` checked for calibration against an ensemble of seeds.
2. **Interface 1, reporting only.** Covariance and status come back through it.
   MaNTA reports `G +- sqrt(w^T Sigma w)` beside `objectiveEstimate()` but
   makes no requests yet. Implementations: `DirectModel` and `FilterProxy`, on
   the toy box.
3. **Error control.** `ObjectiveTolerance`, `ObjectiveBudget`, requests with
   the forcing schedule, the stopping test and its outcomes. The toy box gains a
   run-length knob, `s ~ n^(-1/2)`, with cost counted in effective samples.
   Measured: does the solve meet 1% in W over seeds -- the coverage -- and at
   what cost against a solve at fixed accuracy?
4. **GPProxy**, outside the core, implementing interface 1 over interface 2:
   it answers from its posterior where `s_post <= s_m`, and otherwise samples
   the model where that most reduces `Var(w^T Gamma)`. Measured against
   `FilterProxy` at equal model cost, on the toy, on Shestakov with Parker et
   al.'s noise, and on the T3D ReLU benchmark.
5. **Discretisation error in G**: a goal-oriented estimate (dual-weighted, with
   an enriched or postprocessed adjoint) to fill the discretisation share, and
   to drive `DegreeAdaptation` by G instead of `||u* - u_h||`.

## Open questions

* **The tolerance's form.** Relative to `|G|` by default -- the "1%" -- with an
  absolute floor for an objective near zero? Several objectives (`ng > 1`) each
  with a tolerance: the target at a node is the tightest any objective needs.
* **Where the hooks live.** Virtuals on `TransportSystem` with null defaults,
  or a `NoisyTransportSystem` subclass in C++ bound to Python? The former keeps
  one physics interface. The latter keeps the uncertainty optional in type as
  well as in behaviour.
* **Time-dependent runs.** This design is for steady solves. A transient's
  error in G accumulates over steps, and IDA's local error control is the place
  it would join. Out of scope for now.
* **Uncertainty in the derivatives.** Newton needs J only roughly. The
  significance guard used the derivatives' errors, and a surrogate can supply
  them. Is that part of interface 1, or the system's own business?
* **Field models and scalars** are out of scope: `w` covers the transport
  physics outputs only.
