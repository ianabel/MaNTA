# Physics cases

Registering a case, declaring it as data, boundary kinds, hooks, and the evaluation plan.

`PhysicsCases::map` is a process-global map populated by static-initialisation
side effects: `REGISTER_PHYSICS_HEADER(T)` in the header plus
`REGISTER_PHYSICS_IMPL(T)` in the `.cpp` instantiate a `PhysicsCaseRegister<T>`
whose constructor inserts a factory. Consequences:

* A case only appears if its object file is linked in — nothing references it
  directly, so a missing entry is a link-line problem with no compile error. An
  unknown name now throws from `InstantiateProblem` with the list of what *is*
  registered, rather than returning `nullptr` for every caller to check.
* A **duplicate name throws**. It used to be a bare `map::insert`, so the first
  registration silently won and the second case was unreachable.
* The map is never reset, so tests must use unique throwaway names.
* A case can live **outside this tree**: build it as a shared object against the
  installed headers and name it in the config's `PhysicsPlugins` array, which
  `runManta` dlopens before instantiating. See "Out-of-tree builds" in `build.md`.

**A case declares itself as data.** `TransportSystem`'s only constructor takes a
`SystemSpec` (`SystemSpec.hpp`): the variables, scalars and aux variables with
their names, descriptions, units, per-end `BoundaryCondition` and the differential
flag. `nVars`/`nScalars`/`nAux` are `const` and derived from it; the spec is
validated in the constructor, so a part-built case cannot exist.

**There are three boundary kinds, and the solver only has two booleans.**
`BoundaryKind` is `Dirichlet | Neumann | Mixed`, and a `BoundaryCondition` carries
the coefficients of `a u + b q + d sigma = c` for the last of them (`c` is what
`LowerBoundary`/`UpperBoundary` returns, so the coefficients are data and only the
datum depends on `t`). But `isLower/isUpperBoundaryDirichlet` are the only
interface `SystemSolver` uses, so **every `!isLowerBoundaryDirichlet(var)` means
"Neumann *or* Mixed"** — ask `lowerBoundaryKind` / `isLowerBoundaryMixed` /
`lowerBoundaryCondition` where the difference matters. In the assembly it usually
does not: `effectiveBoundaryCondition` turns a Neumann end into `b = 1` (or
`d = 1` under `zeroFlux`, which is now the flag's only reader), so both kinds go
through one path. Three properties are load-bearing and easy to lose:

* **Neumann *is* `b = 1`, bit for bit.** `zeroFlux` is `d = 1`. That equivalence
  is what licensed reimplementing the flag, and it was checked by building the
  previous binary and diffing the netCDF output byte for byte, because the
  regression suite compares at 1e-2 and no config sets `zeroFlux` at all.
* **Dirichlet is *not* `a = 1`.** It is an identically zero trace row *and*
  column with the datum substituted into the cell rows; `a = 1` is a weakly
  imposed Dirichlet. `validate()` refuses `b = d = 0` for that reason.
* **`a` goes on the `H` diagonal (the lambda column), carrying the outward
  normal** — `-a` below, `+a` above. The HDG literature imposes a boundary
  condition as a relation between the *numerical flux* and the *trace unknown*
  (`refs/HDG-Helmholtz-Robin.pdf` eq. 2.3), not the interior `u`. A sign error
  here converges at the right rate to the wrong function, so it is pinned by
  closed-form tests at each end separately rather than by an order study.

`docs/physics_interface.rst` has the stability constraint on the signs of `a` and
`b`, which is a real restriction rather than advice: the wrong pairing is
anti-dissipative and the run diverges. A case whose shape depends on its
config builds the spec in a static helper: `: TransportSystem(buildSpec(config))`.
`numberedFields/Scalars/Aux` produce the placeholder names
`Var0`/`Scalar0`/`AuxVariable0`, and are for a case whose width comes from its
config and so has no names to give — `MatrixDiffusion`, `MatrixDiffusionTest`
and `LinearDiffSourceTest`. Every other case names its variables, and the netCDF
groups take those names, so nothing reads output by `Var0` any more: the
regression harness finds a variable's group by its holding a `u`, and
`LoadDataToSpline` reads back `getVariableName(i)`.

Two layers sit above `TransportSystem`: cases may implement its virtuals
directly, or derive from `AutodiffTransportSystem` and supply `Flux`/`Source` in
terms of `autodiff` types, which then derives every Jacobian entry automatically.
Its constructor is `(config, grid, SystemSpec)`.

**Derivative out-parameters arrive zeroed.** `State` and `GlobalState` zero
themselves on construction, so a hook assigns only its nonzero entries; an
omitted one is zero rather than uninitialised heap. `State`'s vectors are private; reach them
through `u(i)`/`q(i)`/`sigma(i)`/`sigmaHat(i)`/`phi(i)`/`scalar(i)` for an entry,
bounds-checked under `DEBUG`, or the same names with no argument for the whole
vector (`s.u()`), which is what the autodiff layer builds its RealVectors from.

**Every physics hook exists in two forms**: pointwise (`SigmaFn(i, State, x, t)`)
and batched (`SigmaFn(i, GlobalState, positions, t)`). The batched defaults in
`TransportSystem.hpp` are serial loops over the pointwise version, several under
`#pragma omp parallel for`; a case may override either level.

**The solver tells a case where it will evaluate it, and the list has to stay
complete.** `SystemSolver::evaluationPlan()` builds an `EvaluationPlan`
(`EvaluationPlan.hpp`): one site per kind of evaluation, entry point and cadence,
with the exact abscissae, plus the grid and `k`. `initialize()` offers it to the
case in its unconditional part, *before* `initialiseMatrices` -- which calls `aFn`
on a solver's first run -- through `TransportSystem::deliverEvaluationPlan`, which
calls `prepareEvaluation` **only if the plan differs from the last one that
instance was handed**. A case may compile per batch shape against it, so **a
physics call at a point set the plan does not list is a bug, not a fallback**:
`EvaluationPlanTests.cpp` records every call a case is handed across a matrix of
configurations and fails on any outside the plan, and on any announced site never
used. Adding a physics call means adding its site to `evaluationPlan()`. The
`TauFaces` and `TauNodes` sites are keyed off `tauEvaluatesFaces()` and
`tauEvaluatesNodes()` and nothing else, so a change to where tau evaluates is a
one-line change there.

**A plan depends on the discretisation and configuration, never on the run**, and
that is load-bearing: `==` is plain defaulted equality, and "nothing changed" is
what decides whether a case hears anything and whether a reuse is a regrid. So a
site only some runs reach is listed always, with its count an upper bound --
`MassMatrix` (a solver's first run only), `InitialProjection` (cold starts only),
`InitialCondition`'s `sigma` sweep (skipped on a copied restart). Making a
site conditional on `initialised`, the restart state or anything else a run
moves turns every rerun into a spurious regrid.

**A plan at other points on an instance that already has one is a regrid, and
the case decides whether it survives one.** Other points, not merely a plan that
is not `==`: `EvaluationPlan::evaluatesAtTheSamePointsAs` compares the grid, the
degree, the Superconvergent flag and the distinct point sets. The plan's `steady`
flag is the reason. A Runner arms a steady solve only when `run_ss()` or
`start_steady()` is called, unless `SteadyStateTolerance` is configured, so
`configure()` checks a time march's plan against the steady one the last solve
delivered; the two differ in counts and in the march-only sites, never in their
points, and `==` would refuse a `Fixed` case object every reconfigure after a
steady solve. `SystemSpec::regrid` is `RegridPolicy::Fixed`
by default (a Python attribute cannot be called `None`) or `InPlace`, which takes
the new plan through `prepareEvaluation`. A `Fixed` case is refused a plan at other
points (`deliverEvaluationPlan`, and `PyRunner::configure` for a reconfigured case
object; `PhysicsInstance::usable` for the drivers)
-- unless an adaptation driver may rebuild it. That is the configuration key
`RebuildPhysicsOnRegrid`, a fallback for cases written before plans:
`PhysicsInstance` (`PhysicsInstance.hpp`) holds the caller's case and adjoint
slots and a registry rebuild function, and `solverFor` builds a probe solver,
compares plans, and if needed destroys the case, re-instantiates it by name for the
new grid, moves the restart state across (`copyRestartFrom`) and re-obtains the
adjoint. Order matters: the previous solver must be gone before `solverFor`,
because a rebuild destroys the case it pointed at, and warm-start restart values
are set *after* `solverFor`. Only runManta and a `Runner` built from a case *name*
can rebuild; the drivers refuse a case that can do neither up front, whenever they
*may* change the plan, before the first solve.
Every C++ case in the tree declares `InPlace`; those holding `xL`/`xR` take them
again from each plan (`AutodiffTransportSystem::prepareEvaluation` for the
autodiff cases), so a case added with grid-dependent state has to do the same or
stay `Fixed`. The rebuild path is reached only by the `Fixed` fixtures in
`EvaluationPlanTests.cpp` and `test_evaluation_plan.py`.

**`PhysicsParallelism` costs a level by its plan, so a plan must be buildable for
any level.** `SystemSolver::evaluationPlanFor(grid, k)` is the plan this solver's
configuration would make elsewhere, and `ParallelFill.hpp` counts its recurring
batched sites in rounds of N to *fill* each level a controller chooses (degree up
to `MaxPolynomialDegree`, past `MaxDegreeIncrement`; graded cells; ladder rungs).
So everything `evaluationPlanFor` reads must be the configuration or its
arguments, never the solver's own grid, k or run state;
`a_plan_for_another_level_is_the_plan_a_solver_there_makes` compares it with a
solver built at that level and is what fails if a site starts reading `y`. The
graded-mesh fill keeps the wall cell and caps the layer's ratio at 1/2 because
`gradedMeshPoints` puts boundaries at `layer * ratio^j`: past 1/2 the wall cell's
neighbour is the narrower of the two.

**`MaxPhysicsBatch` is enforced twice, and the second is the one that cannot be
forgotten.** Every function in `ParallelFill.hpp` that chooses a level takes it as
`maxBatch`, and `runAdaptiveDegree` lowers its ceiling with `degreeCeiling`; then
`initialize()` refuses any plan whose `largestRecurringBatch()` exceeds it,
*before* `deliverEvaluationPlan`, so a new route to a level that forgets the cap
throws instead of handing the case a batch it was promised it would not get.
Keep that check ahead of the delivery. The discretisation estimate's batch at the
enriched nodes is larger than the residual's, and goes through the same check.
