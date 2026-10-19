# Error control on an objective — design

**Status:** agreed 2026-10-17; phases 1-4 below, all built. Branch
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

**Chosen: the enriched adjoint, two degrees beyond the scheme's physics
interpolation** (`k + 2`, `k + 3` under `Superconvergent`), with the enriched
residual from the physics at the enriched nodes. Both of the text's assumptions
about the postprocessed candidate failed on measurement -- that it would be
accurate enough, and that the `k+2` star nodes would carry enough physics -- so
the estimate makes two physics calls per steady solve. See Phase 3 as built.

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
  all of them. Both are a number or an array (the `Absolute_tolerance` type):
  one value for every objective, or one per objective, checked against `ng`
  when a solve starts. Defaults `1e-3` and `0`.
* `ObjectiveBudget` -- the shares, as the array `[discretisation, algebraic]`,
  non-negative and summing to 1. Default half and half.
* `SteadyStateTest = "Objective"` -- see Stopping.
* `AdaptationTarget = "Solution" | "Objective"` -- what `DegreeAdaptation` and
  `MeshAdaptation` adapt to. `"Solution"` is today's behaviour and the default.

## Stopping

`SteadyStateTest = "Objective"` stops the steady solve when `|lambda^T F|` is
below the algebraic share for every objective. It is evaluated at each Newton
iteration of the undamped solve, at no physics cost, and not during damped
continuation, where the state is far off and the estimate means little. With
`J^T lambda = dG/dy`, `lambda^T F = (dG/dy) J^-1 F`, which is the objective's
change under the Newton correction the iteration solves for anyway, so it needs
no adjoint solve (see Phase 2 as built). **It controls only the named objectives.**
Anything else the run outputs may be less converged than the `NewtonStep` test
would leave it, and the docs must say so: name every output that matters as an
objective. The adjoint gradients `G_p` are taken at that state too.

## Adaptation

Under `AdaptationTarget = "Objective"`:

* `DegreeAdaptation` replaces its control quantity `||u* - u_h|| / ||u_h||` with
  `|eta_disc| / (eps_disc |G|)`, in the same rule `dk = ceil(log_base(E / eps))`.
  The base is remeasured: functionals typically converge at a higher rate in
  `k` than the solution does.
* `MeshAdaptation` grades only while the sample does not meet the
  discretisation share, and then where the modal decay says. The distribution of
  `|eta_K|` was measured as the reading of where, and is not one: see Phase 4 as
  built.
* The plan machinery (`ParallelFill`, `MaxPhysicsBatch`) is unchanged.

## Exposing the adjoint to Python

The `Runner`, and `FFIRunner` for the host-side reads, gains:

| call | returns | phase |
|---|---|---|
| `solve_adjoint(g=None)` | solves at the current state, on demand -- between slices, not only at the end -- reusing the last build | 1 |
| `adjoint_state(g)` | `z` as nodal arrays per variable: `sigma`, `q`, `u`, `aux`, the traces, the field DOFs (no scalars: see Phase 1 as built) | 1 |
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

## Phase 1 as built

Where the build departs from the text above, or measured something it did not
anticipate:

* **"Reusing the build" means its physics derivatives, not its factorisation.**
  The forward factorisation is of `MX` at the build's alpha (IDA's `cj`, or
  `1/dt` in continuation), and the adjoint wants `dF/dy` at alpha = 0. So every
  `updateMatricesForJacSolve` keeps the derivative blocks it took
  (`jacobianDerivatives`), and `assembleAdjointOperator` builds `M^T` from them
  at alpha = 0 into factorisations of its own. That costs an assembly and `nCells`
  dense factorisations per adjoint solve, and no physics call; it also means an
  adjoint solve between forward ones no longer disturbs them (the field blocks,
  which the adjoint does share, are put back lazily by `solveJacEq`).
* **The end-of-run adjoint is evaluated afresh only when it has to be.** It is
  defined at the final state `y` and time `jt`, where the last build usually is
  not: IDA's last build is at one of its iterates, and a Newton solve stops one
  correction past its last. So `adjointDerivativesAt` reuses the build when it was
  taken at that state bit for bit, which a steady solve that estimated its
  objective guarantees, and otherwise evaluates the derivatives once -- once, not
  once per objective as before. Every existing gradient, adjoint state and
  objective estimate in the three suites and `python-examples/adjoints` is bit for
  bit unchanged (441 hashed values over the suites, 305 over the examples).
* **The current state** is the one `getSolution()` reads: the final state after
  a run, the last slice's between slices. The operator is exact there when the
  last build was there, which `operator_at_state` reports; otherwise it is the
  last iterate's Jacobian.
* **No scalars.** The adjoint operator has never carried the scalar bordering
  (`v`, `w`, `N`): the end-of-run adjoint ignores scalars entirely. The new calls
  refuse `nScalars > 0` rather than return a `z` for a different `J`, and
  `adjoint_state` has no scalar block. The end-of-run path is unchanged and still
  silently scalar-free; that belongs with the field-model note below.
* **`w` covers the flux and the sources**, not the auxiliary constraints, which
  the finite-difference check does not reach.
* **The discrete adjoint converges slowly per node, and that bears on phase 3.**
  `w` matches finite differences of `G` to 2e-12 of its size, so it is the exact
  derivative of the discrete objective. Against the continuum sensitivities
  (`Int z l_m dx`, `-Int z' l_m dx`) the per-node error falls, 8 to 16 cells, at
  orders 2.1, 3.1, 2.0, 3.0 for k = 1..4 without `Superconvergent` and 2.1, 3.0,
  4.0, 3.0 with it -- below the solution's. Part of the reason is that
  `G = sum_m w_m g(u_m)` has gradient `diag(w) dg` rather than the mass-matrix
  Riesz representer, an alternating mean-free difference within each cell that
  `z` inherits: at k = 4, with `z` a quartic the basis represents, `z_h` is off
  by 2.4e-4 of `max |z|`. An estimator built on `z - z_h` has to see past that
  pattern.

## Phase 2 as built

* **The estimate is read off the Newton correction, not solved for.** The text
  above first had an adjoint solve per undamped iteration. Built that way and
  measured, on `AdjointTestProblem` (a C++ autodiff case, `ng = 2`, Newton mode,
  equal iteration counts, best of seven): the objective test cost 36-55% more
  wall time than `NewtonStep`, of which the adjoint assembly, `nCells`
  factorisations and two transposed solves were a quarter to a third of a whole
  Newton iteration each time, and the objective's own `dg` the rest. The same
  number is `(dG/dy) . delta` for the correction `delta` the iteration has just
  solved for with the same build, so the adjoint solve is dropped: the test
  checks `|z^T F|` from `solveAdjointAt` against it at 1e-15 relative, with and
  without `Superconvergent`, and every count and error in the benchmark table is
  identical between the two implementations. What remains per undamped
  iteration is the objective's `dg` and `GFn`: on `AdjointTestProblem` the
  overhead fell to 24-37%, and the rest is `AutodiffAdjointProblem::dg`, which
  differentiates by autodiff at every node and costs about as much as a Newton
  iteration of that cheap case; NumPy objectives on the Python benchmarks are
  lost in the timing noise. `solveAdjointAt` stays the way to
  get `z` itself -- the noisy design's `w`, and phase 3 -- between slices.
* **Where it composes with NewtonStep.** The damped continuation is
  `NewtonStep`'s to the bit; only the undamped solve's convergence test differs.
  The estimate is trusted from the second iteration once the corrections
  contract, and on the first only when that correction is within one unit of
  the error weights -- the gauge the continuation already uses for a state that
  has stopped moving. It is made at the iterate the correction starts from and
  the correction is then applied, so the returned state is a Newton step better
  than the one tested: on the four benchmarks at 1e-2 to 1e-4 the error left in
  `G` is at most 0.36 of the share (Shestakov under PseudoTransient at 1e-2,
  1.8e-3 against 5e-3), and usually orders of magnitude inside it.
* **An unreachable tolerance ends in a new outcome, `Stalled`**, returned like
  `OutOfSteps`: the correction reached round-off in the error weights, or
  `NewtonStep`'s own test passes and the estimate stopped falling (slower than
  `RATEMAX`), with the objective test unmet. An objective at zero with no
  absolute floor is the ordinary way to get one. Without it the undamped solve
  ran to `NewtonMaxIterations`, was rejected, and dropped into damped
  continuation until a budget ran out.
* **What it saves, measured** (`python-examples/objective-tolerance/`): the last
  one or two Newton iterations, and only at a loose tolerance -- Jardin at 1e-2
  4 against 5, two-temperature 3 against 5. From 1e-3 down the two tests stop on
  the same iteration on Park, Jardin and two-temperature. What it buys is the
  accuracy asked for where `NewtonStep` does not deliver it: on Shestakov
  `NewtonStep` at its default leaves the particle inventory 6.5e-4 (Newton) and
  1.8e-3 (PseudoTransient) from the fixed point, and the objective test at 1e-3
  spends two more iterations to return 6.4e-8 and 3.0e-6.
* **Refusals** are where they can first be made: in the configuration,
  `Objective` without `solveAdjoint` or under `TimeMarch`, an objective key under
  any other test, and a budget that is not two shares of one; at the start of a
  solve, a tolerance list neither 1 nor `ng` long, an objective with neither
  tolerance, global scalars and a `Diffusive` tau.

## Phase 3 as built

* **The estimator** (`DiscretisationEstimate.cpp`, `docs/adjoints.rst`
  "The discretisation error"): `eta_disc = z_e . F_e(U~) + [G_h(u_h) - G_e(U~)]
  - (dG/dy) . J^-1 F_h`, where `U~` is the state lifted to the enriched degree
  `k_e` (u* in place of u_h under `Superconvergent`), `F_e` and `z_e` the plain
  degree-`k_e` HDG scheme's residual and adjoint there, and the last term the
  algebraic error `estimateObjective` already subtracts -- so `G ~ corrected -
  eta_disc`. The middle term is the objective's own quadrature. The enriched
  scheme is a second `SystemSolver` that never runs: its `initialiseMatrices`
  skips `aFn`, and the residual's transport rows were split out
  (`assembleTransportResidual`) so it can assemble them from physics values the
  estimate evaluated. `eta_K` shares each face's trace rows between its two
  cells, and the cells sum to the total to round-off.
* **Two degrees, not one.** The plain scheme's `G` converges at `h^{k+2}` for
  even `k` and `h^{k+1}` for odd (the `k+1`-point Chebyshev rule gains a degree
  when `k+1` is odd), so for even `k` degree `k+1` converges no faster and an
  estimate built on it is off by a constant ratio of errors that refinement
  does not remove -- 1.055 at `k = 2`, 1.06 at `k = 4` on the linear
  manufactured problem. `Superconvergent` at degree `k` gives the plain scheme's
  `G` at `k+1` to every printed digit there, so its increment is 3. With an
  increment of 1 under `Superconvergent` the estimate of `int u` and `int x u`
  on the linear problem is zero to round-off for `k >= 2`.
* **Effectivity**, over four manufactured problems x four objectives x `k = 1..5`
  x 4, 8, 16 cells, errors above 1e-11: plain scheme median 1.000, range
  0.84-1.44, 95% within 10%; superconvergent median 1.000, range 0.88-1.00, 99%
  within 10%. The postprocessed adjoint: 0.999, -1.27-5.28, 67% (plain), and
  1.000, 0.73-5.08, 78% (superconvergent) -- rejected as erratic. It fails on
  nonlinear objectives (the `diag(w)` pattern phase 1 measured, inherited by
  `z_h`) and on nonlinear physics. Park: 0.98-1.04 at `k = 1..5` on 2-16 cells,
  both schemes. A fifth problem, the nonlinear flux through an aux variable, is
  within 0.94-1.08 on 16 cells but pre-asymptotic below that, its error changing
  sign between meshes. No calibration.
* **Shestakov is the limit, and an honest one.** 0.75, 0.64, 0.55, 0.49, 0.43
  for `k = 1..5` (plain, 40 cells; 0.91 down to 0.77 superconvergent), steady in
  `h`. Its error is the boundary-cell flux deficit, `~ h / (k+1)^2`, which the
  enriched scheme commits too, so the estimate is `e_k - e_{k_e}` and its
  effectivity `1 - ((k+1)/(k+3))^2` = 0.75, 0.64, 0.56, 0.49, 0.44 -- as
  measured. With `zeroFlux` the axis's `x^{4/3}` still limits the rate, and the
  effectivity is 0.30-1.10 (plain), 1.17-1.91 (superconvergent); with the source
  edge inside a cell it is -3.1 to 3.9. An error that does not decay faster in
  `k` than the scheme's own is not seen.
* **Cost**: one `ComputePhysics` and one `ComputePhysicsDerivatives` at
  `N(k_e + 1)` points per estimate -- a new evaluation-plan kind,
  `DiscretisationEstimate`, and cadence, `PerObjectiveEstimate`, listed whenever
  the solver is configured to estimate (the estimate refuses any other solver,
  so the listing is exact); `MaxPhysicsBatch` applies to it. In wall time on a
  free case, 3.1-4.9x a residual plus a Jacobian build (the degree-`k_e`
  assembly); on the Python benchmarks +7% to +35% of a steady solve.
* **Exposure**: `EstimateDiscretisationError` (needs `solveAdjoint` and
  `EstimateObjectiveOnFinish`, refused under `TimeMarch`; refused at
  `initialize` with scalars, a field model, a non-constant tau, `k = 0`, or a
  batch past `MaxPhysicsBatch`). `ObjectiveEstimate::discretisation` in C++,
  `discretisation`, `discretisation_per_cell`, `x_faces` and `continuum` in
  `Runner.objectiveEstimate()`, and one line per objective at the end of the
  solve. Every existing output is unchanged: the regression configs' `.nc`
  files byte for byte, and `G`, `G_p`, `objectiveEstimate` and the solutions of
  the 32-run Python record bit for bit.

## Phase 4 as built

* **The degree** (`runAdaptiveDegree`, `objectiveAccuracy` in
  `DegreeAdaptation.cpp`): the control quantity is the worst objective's
  `|eta_g| / (beta_disc max(eps_g |G_g|, eps_abs_g))`, with `G_g` the fixed
  point's (`corrected`) and the estimate the one each level's steady solve makes
  on its way out, so measuring a level costs nothing beyond solving it. The rule
  is Giorgiani's, unchanged, with target 1.
* **The base stays 10.** Remeasured on 4 cells: the estimate falls 19-33x per
  degree (geometric mean; Park, the manufactured nonlinear problem with `int u`
  and `int u^2`) against 10-22x for `||u* - u_h||` -- the faster rate the text
  expected -- but in steps of 2.8 to 390, as the Chebyshev rule gains a degree on
  alternate `k`. Over nine runs at 1e-3, 1e-5 and 1e-8 a base of 10 took the
  fewest levels (20; 20 and 30 took 21, 50 took 22, 100 took 23) and the fewest
  physics evaluations, every run meeting its tolerance: an overshoot costs a
  degree, an undershoot a whole solve.
* **The mesh: the objective says whether, the modal decay still says where**
  (`objectiveGradingDecision`, `MeshAdaptation.cpp`). A sample that meets its
  tolerance is not graded; one that does not is graded as the solution target
  grades it. The text above had `|eta_K|`'s distribution decide which end, and
  the measurement says it cannot:
  * the per-cell estimates cancel -- they are the enriched residual tested against
    the whole enriched adjoint, not its part the discretisation cannot represent
    -- so `sum_K |eta_K|` is 3700x `|eta|` on `x - x^{4/3}` at 10 cells and 10^6 x
    on the manufactured sine problem, the end cell and its neighbour on the first
    `+1.00e-3` and `-1.00e-3` against a total of `5.4e-7`;
  * an end region's share of `sum_K |eta_K|` (the end cell and its neighbour)
    separates singular from smooth ends on uniform meshes (0.998 or more against
    0.948 at most, 3-20 cells, `k = 3..6`), where the decay already does from 4
    cells, and not off them: it reached 0.99 at 41 smooth ends of 1217 on graded,
    random and deeply graded meshes, and singular ends the decay reads rough held
    0.04 to 1;
  * as a filter on the decay (a majority share required as well) it met the
    tolerance from 490 non-uniform starts in 347 runs against 353 without.
  This bears on phase 3: `ObjectiveEstimate::discretisation.perCell` sums to the
  total but is not a localisation of the error.
* **What the target implies**, in `loadSolverConfig`: `solveAdjoint` required,
  `EstimateObjectiveOnFinish` kept on, `EstimateDiscretisationError` turned on, the
  `Objective` steady test the default (a named test is kept), a positive
  discretisation share, and the solution target's `DegreeTolerance` refused.
  `MeshAdaptation` keeps its `k >= 3` floor and its decay keys.
* **`ParallelFill` and `MaxPhysicsBatch` are unchanged**, but the estimate's batch
  (`N(k+4)` under the superconvergent scheme the drivers use) is not a recurring
  one and so not counted by them, while `initialize()` refuses a level whose
  estimate breaks the cap. The objective target therefore holds its degree
  ceiling and a filled graded cell count to levels whose estimate fits
  (`discretisationEstimateFits`).
* **`Runner.G()` under `Superconvergent` read `u_h`** while
  `getAdjointGradients()`, `objectiveEstimate()`, the objective test and
  `WriteAdjoints` all read `u*`; it now reads `u*` too. Before, the two disagreed
  in 9 of 16 superconvergent runs of the phase-2 record, by up to 7.3e-6 relative;
  nothing else moved.
* **Measured** (`python-examples/objective-tolerance/adaptation.py`,
  `docs/adaptivity.rst`): against closed forms at 1e-3 and 1e-5 the objective
  target meets `eps_G` wherever it can be met; the solution target with its
  `DegreeTolerance` tuned on the closed form matches it, at 1e-1 to 1e-4 depending
  on the row, for 1.2-1.7x fewer physics evaluations where both take the same
  degrees -- the estimate's four sweeps per level. Against the default
  `DegreeTolerance` it spends fewer in seven runs of ten. From uniform meshes,
  210 runs: 176 converged within `eps_G` (closest 0.81 of it), none outside, 34 at
  the ceiling with a warning, all on meshes the solution target also ends on with
  the same error. From 35 non-uniform meshes, 350 runs: 232 converged within, one
  outside -- a singular wall cell where the estimate has the wrong sign and is 30x
  short -- and 117 at the ceiling, all on the singular problems. **The adjoint's
  accuracy** (the open question below) does not limit it there; the estimate's
  underestimate of errors that decay only algebraically in `k` does, and the
  budget's other half is what absorbs it, all but once.

## Open questions

* **The adjoint's own accuracy.** Each estimate uses `z` at the current state,
  whose error is first order in the state's error. Stopping can tolerate that,
  and so, measured in phase 4, can adaptation: one converged run of 560 missed
  its tolerance, and that through the estimate on a singular cell, not the
  adjoint.
* **Time-dependent runs** are out of scope. An adjoint for a transient
  objective is a backward-in-time solve MaNTA does not have.
* **Field models and scalars.** `w` and `eta_disc` cover the transport physics
  first. The field notes record that the coupled adjoint lacks a `dg/dgeometry`
  hook, so a `G` that reads geometry gets `G_field = 0`.
