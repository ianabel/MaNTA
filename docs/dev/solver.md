# The solve path

How a run is driven, how steady solves work, bounds, and how restarts and initial conditions are built. `CLAUDE.md` has the summary.

## Sources that read `du/dt`

**A source may read `du/dt`, and only `du/dt`.** It reaches a case as
`State::udot(j)`, filled by the solver from the `dYdt` vector, and is **zero
unless some variable's `FieldSpec` sets `sourceReadsTimeDerivatives`** — which is
what keeps every case that does not declare it identical to what it was, down to
not allocating the rows. The derivative is `dSources_dudot`, and it enters the
Jacobian as `- alpha * dS/d(udot)` beside the mass term in the `u` block
(`assembleCellMatrix`). Four consequences, each of which has a test:

* **It is invisible at `alpha = 0`.** A finite-difference check of the Jacobian
  written only at the steady point passes with the term deleted, which is why
  `TimeDerivativeSourceTests.cpp` carries both an `alpha != 0` case and an
  `alpha = 0` one that records the blindness rather than relying on it. The
  adjoint is built at `alpha = 0` and is therefore unaffected by this feature
  entirely.
* **It changes what multiplies the time derivative**, from `X` to
  `X - dS/d(udot)`, which is no longer diagonal in the variable index and which a
  case can make *singular* by writing `a_i d_t u_i` into its own source. That
  raises the index of the system; `checkEffectiveMassMatrix` assembles the
  operator at the initial state and refuses the run by name, in the same place
  and for the same reason as the differential-field-DOF check beside it.
* **A steady solve sees `udot = 0`** — `PseudoTransient`'s damping term vanishes
  at its fixed point and `Newton` never has one — so "steady state" means
  `d/dt = 0` inside the sources too, which is the right reading.
* **There is deliberately no `qdot`, `sigmadot` or `phidot`, and the reason is
  `IDASetId` rather than availability.** IDA supplies `y'` for every component,
  algebraic ones included, and `variableTimeDerivatives` already builds a whole
  `GlobalState` from `dYdt` and keeps only its `Variable()`. What makes those
  entries unusable is what the solver has *declared*: `id` marks `u` differential
  and nothing else, so a residual reading `q_dot` would put `alpha`-weighted
  entries in the `q` columns of `dF/dy'` and contradict it. They are also exactly
  zero at the initial point — `setInitialConditions` fills only the differential
  rows and `IDA_YA_YDP_INIT` holds the algebraic `y'` fixed — so a term reading
  one would be wrong for precisely the evaluations `IDACalcIC` converges on, and
  right afterwards.

`docs/superpowers/specs/2026-09-21-time-derivative-sources-design.md` is the
design, including the two phases not built: geometry time derivatives (`dV'/dt`,
`dpsi/dt`) and the field rows.

## Solve path

`Solver.cpp:runSolver` is the driver. It is three phases, which can also be
called separately:

* `SystemSolver::initialize` — allocate the SUNDIALS objects, build the initial
  condition, open the output files, and *sometimes* run `IDACalcIC`. Two
  independent reasons not to:

  * **A steady solve skips it outright.** `solveSteadyState` drives the whole
    residual to zero from `Y` with KINSOL, so a correction made first is
    discarded by the first accepted continuation step. The gate is
    `solvesForSteadyState()`, so `SteadyStateSolver = "TimeMarch"` and a plain
    transient both keep the call.
  * **A restart skips it too, on the copy path.** It resumes from a state the
    previous run had already driven onto the constraint manifold, and `IDACalcIC`
    cannot find that out cheaply: its convergence test is on the Newton *step*, so
    `IDANewtonIC` calls `lsolve` and only then tests `||J^-1 F|| <= epsNewt`
    (`ida_ic.c:404-417`), `IDAnlsIC` calls `lsetup` unconditionally first
    (`ida_ic.c:345`), and the outer loop runs the whole thing twice on success
    (`ida_ic.c:232`). Measured floor, handed a state it had just converged to:
    **2 residual evaluations, 2 Jacobian builds, 2 Jacobian solves, 0 Newton
    iterations** — every time, and on an already consistent `AuxVarTest` warm
    start at rtol 1e-6 that floor is the whole of the saving: 2 residual
    evaluations of 89 and 2 builds of 21.

    **Only the copy path.** A restart onto a *different* degree is projected —
    `setInitialConditions` transfers `u`, `q`, aux and the scalars and then
    rebuilds `sigma` and the trace — so what it hands IDA is a guess like any
    other, and skipping there is a broken run rather than a saving: `AuxVarTest`
    resuming at a lower degree fails with `IDA_ERR_FAIL` when `IDACalcIC` is
    skipped and completes when it runs. `restartWasProjected` carries that from
    `setInitialConditions` to `initialize`, and
    `only_a_copied_restart_is_treated_as_already_consistent` pins both halves.

  **A cold time-marching run always runs it, and there is no option to turn that
  off.** Its guess is not a consistent state, `IDA_ERR_FAIL` on the first step is
  what starting from one looks like, and a caller who does not care about the
  transient wants `SteadyStateSolver = PseudoTransient` or `Newton` rather than an
  uncorrected time march. `ForceConsistentIC` therefore only ever *adds*
  `IDACalcIC` back — to a steady solve, or to a restart that is not as consistent
  as a restart is supposed to be. `initialize` evaluates the residual once and
  reports its WRMS norm through `getInitialResidualNorm()` on every time-marching
  run, armed or not, so a caller can see how consistent the state they resumed
  from actually was; `initialConditionWasCorrected()` says whether CalcIC ran.

    **The decision is made from what the run *is*, not from a residual threshold,
    and that is a measurement.** What `IDACalcIC` tests is
    `||J^-1 F||_wrms` — a *correction to y* — and `weightedResidualNorm` says as
    much. The two differ by the per-row amplification `s_i = ||J^-1 e_i||_wrms`,
    and `s_i` is nowhere near proportional to the `ewt` the norm applies. Measured
    per block, as `s_i / ewt_i`, on three cases (`LinearDiffusion`, `MatTest`,
    `AuxVarTest`):

    | block | `s/ewt` | reading |
    |---|---|---|
    | `sigma`, `q` | 0.6 – 10 | about right, and uniform, so harmless |
    | `u` | 2.3e-4 – 2.0 | over-weighted up to ~4000x |
    | `lambda`, Dirichlet ends | **exactly 0** | weight 1e5–1e8 on rows nothing can reach |
    | `aux` | 0.9 – 39 | under-weighted up to ~10x *relative to `sigma`* |

    The `u` rows are the differential ones, whose residual IDA absorbs into `u'`;
    the Dirichlet trace rows are the ones `residual` never writes, so `J^-1 e_i`
    is identically zero there and they can only dilute the mean. The `aux` row's
    spread is the one that bites, because it is a *relative* error against the
    block a corrected state's residual lands in.

    **What that costs, on the tree as it stands, is calibration.** Across six
    `AuxVarTest` warm-start states — three tolerances, corrected and not — the
    amplification `||J^-1 F|| / ||F||` runs from **15 to 187**. One problem, one
    discretisation. A threshold on `||F||` therefore means something different at
    each of them, which is what makes "set it per problem" honest advice rather
    than a hedge.

    So there is no number to pick, which is why the decision is made from what
    the run *is*. A test that would be correct — `||J^-1 F||` itself — costs one
    residual, one Jacobian build and one Jacobian solve against `IDACalcIC`'s floor
    of two of each: a factor of two, not a near-free test, and not obviously worth
    having when the run already knows whether it is a steady solve or a copied
    restart.

  On either skip the `t0` output slice sees the guess `setInitialConditions`
  built, which is the state the run really started from.

* `SystemSolver::integrate(tFinal)` — the time loop, then the adjoint solve and
  the final netCDF / restart output.
* `SystemSolver::destroySundials` — free all of it. Idempotent, and safe with no
  preceding `initialize`, which is what lets `runSolver` free on both the normal
  and the exceptional path.

**A steady solve can also be taken in slices.** `MaxContinuationSteps` (default
200) bounds one `solveSteadyState`; running out of it is a budget exhaustion, not
a failure of method, and `continueSteadyState()` resumes from the state *and* the
pseudo-time step SER climbed to. `integrate()`'s tail is factored as
`writeSteadyState()` + `finishRun()` so a sliced solve ends the same way an
unsliced one does; `finishRun` is shared with the time-marching branch and closes
the output files, so it runs once per run. A second `solveSteadyState()` would resume from
neither — `SteadyState.cpp` re-enters at `PseudoTransientInitialStep` unless the
`resume` flag says otherwise — and re-climbing the ramp is the whole solve rather
than a margin on it: a `NonlinearDiffusion` needing 15 continuation steps takes
the same 15 in slices of three when each resumes, and does not converge in 40
slices when each starts over
(`a_resumed_steady_solve_does_not_re_climb_the_ser_ramp`). Slicing requires
driving the phases directly, because `runSolver` frees the state on its way out
of a failed solve, so `PyRunner::run_ss()` cannot do it.

**What a steady solve spends on physics is counted in whole grid sweeps, and the
budget is exact rather than approximate**: `2 + 3n` for `Newton` and `2 + 4n`
for `PseudoTransient`, `n` being continuation steps, pinned by
`a_steady_solve_spends_the_physics_sweeps_it_has_to_and_no_others`. The two
fixed sweeps are `assignSigmaFromFlux` building `sigma` from the initial condition and
the merit function's `||F||` at the initial state; each step is then KINSOL's
residual at both ends plus one Jacobian, and a step at *finite* `dt` costs a
fourth. Two properties keep it there, and both are the kind that would be lost
silently, since a duplicate sweep changes no answer and only shows up on the
bill of a case whose flux is expensive:

* **`setInitialConditions` returns early on a steady solve**, before solving the
  initial `du/dt` out of the u row. That derivative reaches nobody there --
  `solveSteadyState` damps through its own zeroed `ptcDYdt`, and on convergence
  it overwrites `dYdt` with zero, since the defining property of the answer is
  that `dy/dt` vanishes. The gate is `solvesForSteadyState()`, so `TimeMarch`
  keeps it: that path reaches a steady state through IDA, which wants a
  consistent `y'` at `t0` like any transient.
* **`steadyResidual` records its own norm when `dt` is infinite**, and the loop
  reads that instead of calling `steadyNorm()` again. At `dt = inf` the damping
  term is identically zero, so the function KINSOL evaluates *is* the steady
  residual, at the same state -- it is recorded rather than recomputed so the
  number is bit for bit what `steadyNorm()` would have returned, which is what
  keeps the SER schedule and every run's step sequence unchanged. A stamp
  counter guards it: a `KINSol` that made no successful steady-mode evaluation
  leaves the stamp where the loop's snapshot found it, and the loop evaluates
  for itself. Never consulted at finite `dt`, where the damped residual is a
  different function and any small enough `dt` makes it small.

**All of the above is the `SteadyStateTest = "Residual"` path; the default,
`"NewtonStep"`, is a second loop** (`solveSteadyStateNewtonStep`, `SteadyState.cpp`)
that `solveSteadyState` dispatches to before touching KINSOL. A test of the KINSOL
path's mechanics -- its sweep budget, its counters, the ||F||-based rejection --
must therefore name `Residual`, as the ones in `SolverLifecycleTests.cpp` do.
It is IDA's corrector test,
`ss * ||J0^-1 F||_WRMS(ewt) <= SteadyStateStepTolerance`, on the undamped Newton
correction, with `SUNNonlinSol_Newton` as the inner solver because KINSOL has no
hook for a convergence test of the caller's own. Its budget is `1 + n` flux
sweeps and `n` derivative sweeps over `n` Newton iterations in `Newton` mode,
pinned by `a_newton_step_solve_spends_one_flux_and_one_derivative_sweep_per_iteration`;
no merit evaluation exists to be duplicated. A damped continuation step is a
single Newton iteration, accepted as it stands (Kelley and Keyes), halved when
its correction reverses the last one -- the signature of a kink in the flux, which
no `dt` damps because sigma and q are algebraic -- and only the undamped solve
that ends the run is iterated and tested. Because a damped step is accepted
before its result's residual is evaluated, the loop keeps where the last one
started (`nsLastGood`) and undoes it when the next step finds no finite
residual there or the case throws there; retrying in place evaluates the same
residual at any `dt`, and is what it used to do
(`a_damped_step_onto_a_state_with_no_residual_is_undone`). A step whose result
breaches a variable's bound is not taken in the first place: `steadyNewtonSolve`
checks the result before acceptance, puts `Y` back and returns it unconverged
(`nlsBoundRejected`, per step, not carried), so the undo is never needed for a
bound -- relying on it costs 14 continuation steps on the saturating-sink
fixture where checking first costs 11 (see "Bounds" below). Everything it carries
between steps (`ptcStep`, `nsFprev`, `nsNextUndampedDt`, `nsUndampedCorrectionLimit`, `nsUndampedNext`,
`nsLambda`, `nsCalmSteps`, `nsHaveDprev`, the vector `nlsDprev` and the undo
state `nsLastGood`, `nsHaveLastGood`, `nsLastGoodDt`, `nsLastGoodFprev`) is a member, so a
sliced solve resumes bit for bit, and `a_sliced_newton_step_solve_resumes_exactly`
pins that at zero tolerance.
`docs/superpowers/specs/2026-10-12-kinsol-convergence-test-hook-design.md` is the
upstream KINSOL change that would let this path return to KINSOL.

**`SteadyStateTest = "Objective"` is the same loop with a different test**: the
undamped solve stops when, for every objective, `|z^T F|` -- the algebraic error
in G -- is below the algebraic share of `ObjectiveTolerance` (`ObjectiveBudget`,
with `ObjectiveAbsoluteTolerance` as the floor). **It makes no adjoint solve.**
Since `J^T z = dG/dy`, `z^T F = (dG/dy) . J^-1 F`, and `J^-1 F` is the Newton
correction the iteration has just computed from the same build; an adjoint solve
per iteration measured 36-55% slower for the same number to 1e-15
(`the_estimate_is_the_adjoint_one` pins the agreement). The estimate is trusted
from the second iteration once the corrections contract, and at the first only
when that correction is within one error-weight unit. A tolerance the solve
cannot resolve returns `SteadyOutcome::Stalled` rather than spinning. It controls
**only the named objectives** -- anything else may be less converged than
`NewtonStep` would leave it -- and it needs `solveAdjoint`. The objective keys
are read by this test and by `AdaptationTarget = "Objective"`, and refused under
neither rather than ignored. `ObjectiveSteadyTests.cpp`
pins its threshold, its sweep budget and exact resumption of a sliced solve.

**The residual at the initial state is not one of the removable ones**, despite
duplicating KINSOL's first evaluation: it is also the already-converged test,
the early return that makes a warm start cost one sweep rather than a Newton
solve. And KINSOL offers no way to hand it a residual it did not compute, so
three sweeps -- two residuals and a Jacobian -- is the floor for a Newton method
on a problem it does not know is linear. `PERFORMANCE.md` has the comparison
against a direct solve, which pays one.

Every SUNDIALS handle is a member, not a local, so those three can be split.
`ctx` is the exception: it belongs to the `SystemSolver`, not to a run, and
`destroySundials` must not touch it.

**IDA's error weights are `ErrorWeights = "AlgebraicMax"` by default, not
pointwise**, and that is a measurement (`docs/running.rst`, "Error weights", has
the table). `getErrorWeights` gives every unknown `1/(rtol |y_i| + tau_i)`;
the default floors `tau` for sigma, q and aux at `rtol` times the variable's
largest magnitude on the grid, where `Pointwise` uses `atol` everywhere. Under
`Pointwise`, sigma and q next to a Dirichlet wall -- crowded Chebyshev nodes, a
stiff diffusive mode, backward Euler lagging its slow manifold by ~h dg/dt --
were held to ~atol absolutely, and IDA read that lag as local error that jumps
when h doubles: an order-one limit cycle, ~25000 steps for ~200, on the coupled
theta pinch, under *both* field solves and at *any* FieldSolveTolerance. Which
runs fell in was set by round-off, so it showed up as a CI leg timing out on
one compiler and BLAS and not another -- and a green run proved nothing. The
weights are also the norm of IDA's Newton test, the field sweep's stopping test
and the NewtonStep steady test, so a change here moves all four.
`SuppressAlgebraicError` is separate and composes with any scheme. **It is on by
default for a run with a field** -- a `FieldModel` or an `EquilibriumModel` --
and off otherwise; the config key is a `std::optional` read by presence and the
solver resolves it (`suppressesAlgebraicError`), since only the solver knows by
`initialize()` whether a field is attached. With the algebraic rows in IDA's
error test the coupled theta pinch falls into an order-2 trap or does not as
round-off decides (149-290 steps over a +-8% tolerance sweep, 564 on one CI
runner); suppressed it takes 99-101 at every tolerance. Without a field the
trade goes the other way: its error stops following the tolerance on the
nonlinear-flux configs, and a restart reads `sigma` and `q` back from the file,
so `LinearDiffusion`'s restarted `q` lands 100x further from the uninterrupted
run. `docs/running.rst` has the tables.

**The three can be run again on the same object**, and a reused solver gives the
same answer as a fresh one *bit for bit*. That is pinned by
`a_second_integration_on_one_solver_matches_a_fresh_one`, at exactly zero
tolerance, and the tolerance is the point: the last defect here left the second
run completing, plausible, and wrong in the eleventh digit. Anything that reuses
a solver — a `PyRunner` that stopped rebuilding one, an optimisation driver — is
resting on that test, so do not relax it to something approximate. Note that
`initialize` skips `initialiseMatrices` when `initialised` is already set, so
anything that function computes *once* must either be genuinely run-independent or
be refreshed per run; getting that wrong is what broke reuse before (see the
`RF_cellwise` trap in `CLAUDE.md`).

IDA is handed a **custom `SUNLinearSolver`** (`SunLinSolWrapper`) plus a
**deliberately empty `SUNMatrix`** (`SunMatrixWrapper`) whose only job is to
convince IDA it has a matrix-based direct solver; the Jacobian is never
assembled.

* `SystemSolver::residual` — evaluates the whole residual. Does **not** write the
  Dirichlet boundary rows; those constraints are imposed inside the linear solve,
  so a finite-differenced Jacobian is rank-deficient by exactly the number of
  Dirichlet boundaries.
* `SystemSolver::updateMatricesForJacSolve` — builds and factorises the per-cell
  `MX` blocks from `Matrices.cpp`.
* `SystemSolver::solveHDGJac` — static condensation onto `lambda`, then
  back-substitution.
* `SystemSolver::solveJacEq` — wraps that in a bordered elimination for the
  global scalars: a Schur complement `(N + W^T A^-1 V) mu = s - W^T A^-1 r`,
  then back-substitution. It costs `nScalars + 1` transport solves. **It never
  inverts `N` on its own, and must not.** A differential scalar whose row reads
  itself only through its own `d/dt` (an integral controller's `dH/dt - E`)
  has `alpha` on its diagonal of `N` and nothing else, so `N` is exactly
  singular at an undamped steady step and nearly so at a large pseudo-time
  step, while the bordered system stays well posed. A Woodbury form, which
  inverts `N`, returns NaN there, and the NewtonStep test then never passes.
  `the_bordered_solve_does_not_invert_n_on_its_own` is the guard.

Because the Jacobian is never formed, **an error in it does not produce a wrong
answer — only slow Newton convergence**. That is why several defects in this area
survived a passing regression suite for months, and why the tests that matter here
are `SolveJacTests.cpp` (finite-difference the residual, require `J dy = g`) and
`MMSConvergenceTests.cpp` (observed order of accuracy).

**When IDA does give up, get its own account of the step before theorising.**
SUNDIALS is built here with `SUNDIALS_LOGGING_LEVEL 4`, and `SUNContext_Create`
builds its logger from the environment (`sundials_context.c` calls
`SUNLogger_CreateFromEnv`), so

```sh
SUNLOGGER_INFO_FILENAME=/tmp/ida-info.log SUNLOGGER_DEBUG_FILENAME=/tmp/ida-debug.log build/MaNTA foo.conf
```

yields IDA's per-attempt record — step size, order, `dsm`, and whether the attempt
died in the error test or the nonlinear solve — for no code change at all. Write it
to a file rather than to `stderr` if a test is running, because `CapturedOutput`
redirects the standard descriptors.

Two return codes worth being able to read without looking them up:

* **`IDA_ERR_FAIL` (-3) on the *first* step means the initial condition violates
  the algebraic constraints.** It takes ten error-test failures in one step attempt
  to produce it, and each one cuts `h`; a local error estimate that will not shrink
  with `h` is the signature of an inconsistent state, when `IDASetSuppressAlg`
  is off and so `sigma`, `q`, `lambda` and `phi` are all in the error test. Look
  at what `IDACalcIC` was given, not at the time loop. A `dsm` that *does* fall
  in proportion to `h` and runs into `MinStepSize` is a different thing: an
  algebraic `y'` that is wrong -- the predictor's error in those rows goes like
  `h |y'|`, first order where the test expects second. `initialize()` solves the
  algebraic `y'` exactly after `IDACalcIC` (below), so look at the case's
  explicit time derivatives: a warning that they disagree with the residual
  means a `d..._dt` hook is missing.
* **`IDA_CONV_FAIL` (-4) from `IDACalcIC`** is the same problem one stage earlier:
  the Newton/linesearch could not reach a consistent state from the guess
  `setInitialConditions` built. The guess is worth suspecting before the solver is.
* **`IDA_ILL_INPUT` (-22) with "tout1 too close to t0" is not about the state at
  all** — it is `initialize` handing `IDACalcIC` the wrong `tout1`. That argument
  is an absolute *time*, "the first value of t at which a solution will be
  requested", and it used to be passed the *interval* `dt0 > 0 ? dt0 : dt`. The
  two agree only at `t0 = 0`, which is where every fixture in the tree starts, so
  `t_initial = delta_t` would make `tout1` land exactly on `t0` and kill the run,
  so it is `t0 + (dt0 > 0 ? dt0 : dt)` — the first time `integrate()` asks for —
  and `initialize_starts_at_a_nonzero_time` pins that. Worth knowing because the
  symptom *looks* like a hard initial condition and is not: it reproduces at every
  tolerance, and a warm start that hits it converges in 3 residual evaluations
  once `tout1` is right.
* **`IDA_LINESEARCH_FAIL` (-13) from `IDACalcIC` means some residual row cannot be
  reduced *at all*, which is usually a declaration error rather than a bad guess.**
  `IDA_YA_YDP_INIT` solves for algebraic *values* and differential *derivatives*,
  so it holds every differential value fixed. A row whose only unknowns are
  differential values is therefore a constant, no Newton direction touches it, and
  the backtracking loop runs to exhaustion. The way to get one is to declare a
  scalar differential whose `ScalarG` contains no time derivative: the constraint
  is then algebraic in a quantity `CalcIC` has frozen. That is exactly what kept
  `python-physics/mirror-plasma`'s voltage controller from ever starting — and the
  C++ `MirrorPlasma` had the same misdeclaration, so neither implementation ever
  ran with `useConstantVoltage`. **Before theorising about the guess, ask which
  unknowns each failing row can actually reach** -- `checkScalarDerivative`
  (`Tests/UnitTests/ScalarDerivativeCheck.hpp`) asks it of every scalar row it is
  handed, through `scalarRowsCalcICCannotReach`. A row reaches CalcIC if it
  touches a `d/dt` of `u` or of a differential scalar, or the value of `sigma`,
  `q`, an aux variable or an algebraic scalar. `ScalarTestLD3`'s `G_0 = E - (M0 - M)`
  did not while E was declared differential; E is a function of `u`, so it is
  algebraic, and the `dE/dt` its controller reads is `-dM/dt`, a `du/dt` term.
  A scalar whose `d/dt` some row wants is not thereby differential. The trap is that the
  residual can be tiny — 4.3e-6 there, just the difference between two quadrature
  rules for the same integral — and still fatal, because irreducible beats small.

## Bounds on a case's variables (`FieldSpec::bound`)

A variable may be declared `VariableBound::Positive` (`u > eps`, eps default 0)
or `NonNegative` (`u >= -eps`, eps default the variable's `AbsoluteTolerance`,
and the physics handed `max(u, 0)`); `manta.Field(bound=manta.Bound.Positive,
bound_epsilon=...)` from Python. `Bounds.cpp` is the whole mechanism and
`docs/running.rst` "Bounded variables" what each solve does. What to know before
touching it:

* **The invariant is "never evaluated, never accepted", and it takes two
  checks per solver.** A residual callback refuses a breaching state as a
  recoverable failure before the physics is called -- `static_residual` (IDA,
  `IDACalcIC`) and `steadyResidual` (KINSOL, the NewtonStep loop) -- and every
  acceptance point checks the result: `steadyNewtonSolve` after forming `Y`;
  the KINSOL loop, which also turns `KIN_REPTD_SYSFUNC_ERR` into a rejected step
  *only* when the bound count moved during that call (otherwise it stays the
  fatal error it always was); and IDA through `idaBoundedNLS`, SUNDIALS' Newton
  wrapped so a converged corrector whose `y0 + ycor` breaches is
  `SUN_NLS_CONV_RECVR`. IDA never evaluates its converged corrector, so the
  residual check alone cannot cover it -- though in every time-march fixture in
  `VariableBoundTests.cpp` the residual callback refuses the breaching iterate
  first and the wrapper never fires in a run, which is why it is tested directly
  (`the_corrector_is_checked_before_ida_accepts_it`).
* **`IDASetConstraints` is deliberately not used**, and `Solver.cpp`'s
  `constraints` vector is allocated and never handed to IDA. IDA's constraints
  bound only at zero, check the coefficients rather than the `u*` the
  superconvergent scheme evaluates, and *project* a small violation (one within
  the Newton tolerance) onto zero, or to `0.2 (rtol |y| + atol)` above it for a
  strict bound, rather than refuse it: the run carries on from a state nobody
  computed, which is what refusing exists to avoid.
* **Checked where the physics reads u**: the k+1 basis-node coefficients, or
  `u* = B11 q + B12 u` at the star nodes, formed in `boundViolation` itself
  rather than through `postprocessor->computeUStar`, whose cache the residual and
  the output own. `BoundCheck = "WholeCell"` adds the exact minimum of each
  cell's interpolant of those values (`util/PolynomialMinimum.hpp`: endpoints
  plus the real parts of the derivative's colleague-matrix eigenvalues, clipped
  into the cell, which can only add candidates, never miss one).
* **The clamp lives in `physicsNodesAt`, with its derivative beside every
  `ComputePhysicsDerivatives` on those nodes.** `zeroClampedDerivatives` runs in
  `evaluatePhysicsDerivatives` and `adjointDerivativesAt`; `dSources_dScalars_Mat`
  builds its `State` from `evalOnNode` and clamps it itself. A new derivative
  site on physics nodes needs the same call, or its Jacobian is that of an
  unclamped residual -- `the_clamp_jacobian_matches_a_finite_difference_of_its_residual`
  goes from 1.7e-10 to 3e-2 without it. The objective's two evaluations pass
  `clamp = false`: an objective reads the state as it is, as `GFn` does.
  `clampedVars` is filled in the *constructor*, so the discretisation
  estimate's `enrichedSolver`, which never initialises, clamps too.
  `NonNegative` is refused with `tauScaling = "Diffusive"` (`resolveBounds`): the
  face kappa and the nodal `d tau / dy` differencing do not see the clamp.
* **Inert when unused, checked by `cmp`.** Every entry point returns on an
  empty `boundedVars` / `clampedVars`, and no wrapper is attached. Re-run the
  regression-config byte comparison after touching any of it.

## Restarts and initial conditions

**A restart used to have its trace thrown away.** `setInitialConditions`
finished every restart with `EvaluateLambda()`, which sets `lambda` to
`{{u}}` -- the average of the two cell traces (`DGSoln.hpp`) -- and that is not
the equation `lambda` solves. The HDG trace row is
`Csigma sigma + Cq q + G_c u + H lambda = L(t)`, so applying the average to a
file that already holds a converged trace replaces it with something that
solves nothing: measured on a `TestDiffusion` round trip at
`AbsoluteTolerance = 1e-8`, that one call takes the weighted residual from
2.6e-3 to 556. It is why a restart needed roughly ten times as many residual
evaluations inside `IDACalcIC` as a cold start. What changed is *not* the order
of the two calls -- `ApplyDirichletBCs` still runs first, and the comment at the
site says why -- but that `EvaluateLambda` became conditional on
`!sameDiscretisation && !sameGrid`. So the copy path and a degree projection
over the same mesh never reach it and keep the datum, while a restart onto a
different mesh still runs it and behaves exactly like a cold start.

**The trace is kept whenever the *mesh* matches, not only the discretisation.**
`lambda` has no polynomial degree — `DGSoln::Map` gives it `nCells + 1` entries
and no basis — so a change of degree leaves it transferable verbatim even though
`copy()` refuses the state as a whole. That matters beyond `lambda` itself,
because the `q` row carries a `<lambda, v n>` term: on a `LinearDiffusion`
restart coarsened from `k = 4` to `k = 3` at `atol = 1e-10`, keeping the trace
takes the `q` block from 7.3e7 to 3.2e-7. Only a genuine remesh rebuilds it.

**A Dirichlet trace entry is written by hand or it is wrong, and it has to be
rewritten every time the state is reported.** Its row *and column* in
`K_global` are identically zero, `imposeDirichletTraceRows` pins the correction
to zero and `residual` never writes the row, so it is an unknown appearing in no
equation: nothing in the integration can move it, and it keeps whatever was last
stored there. `setInitialConditions` seeds it; that used to be the only write, so
a run with **time-dependent** Dirichlet data reported `g_D(t0)` for ever. Measured
on `MatTest`, whose `g_D` decays like `exp(-t pi^2 / 4)`: at `t = 0.5` the restart
file's lower-face trace held 0.99999993 against a datum of 0.29121, with the first
interior trace node at 0.28962 -- discontinuous from its own neighbour by a factor
of three.

`ApplyDirichletBCs(y, t)` is therefore called at each point the state is settled
and about to be read: after `IDASolve` returns, at the top of `writeSteadyState()`,
and on the steady-solve failure path. One call covers every reader, because `y`
aliases the `N_Vector` `IDASolve` writes its output into and the netCDF slice, the
`.dat` files, the restart file's DOF vector and `yJac` all read that memory. **It
reaches no equation**: `IDASolve` treats `Y` as an output and resumes from its own
internal state, so a write between calls is discarded, and the cell rows take the
datum from `RF_cellwise` as they always have -- `MatTest`'s `.nc` is byte
identical across the change, its restart DOF vector bit identical on all 388
entries that are not one of the four trace DOFs, at the same 25 residuals and 11
Jacobian builds.

Note the time argument. `ApplyDirichletBCs` used to read the member `t`, which is
assigned `t0` in `setInitialConditions` and never moves again, so a call added
without it would have written `g_D(t0)` and looked like it worked. Any new
reporting path needs the same call; `the_reported_dirichlet_trace_follows_the_
boundary_datum` and the ladder equality test are what would notice one that
forgot.

**The two *seeds* still differ, deliberately.** A cold start runs
`ApplyDirichletBCs` and then an unconditional `EvaluateLambda`, which sets every
trace entry to `{{u}}` and so overwrites the datum; a restart that keeps its
mesh skips `EvaluateLambda` and keeps it. So a cold steady solve of
`LinearDiffusion` at k = 3 on eight cells seeds 0.9999991 at a face whose datum
is 1, where a degree ladder seeds 1 -- identically at every tolerance, since
nothing moves that entry. Making the cold path match was tried and reverted:
`systemsolver_init_tests` holds the initial trace to the initial function at
the faces, and its `exp(-25x^2)` is 1 at an axis whose datum is 0, so pinning
the face would hand `IDACalcIC` a jump of 0.9 where `{{u}}` is smooth. Both
are defensible *guesses* for an initial condition that breaks its own boundary
condition, and since every reported state carries the datum, no reader sees
the difference.

**`sigma` is loaded on a copy-path restart, not recomputed, and that is a
measurement too.** `DGSoln::copy` brings `sigma` across with everything else and
`ApplyDirichletBCs` touches only `lambda`, so `assignSigmaFromFlux` would rebuild it
from bit-identical inputs — at the price of a full `ComputePhysics` over every
node, which is *exactly one residual evaluation's worth of physics*
(`residual` makes the same call, `SystemSolver.cpp:1329`). It also evaluates
`Sources` for every variable and `AuxG` for every auxiliary one and drops both,
since only the flux is used. On a copy-path restart, which now skips
`IDACalcIC` entirely, that was half of all the physics `initialize()` did.

The trade is real and small: a rebuilt `sigma` satisfies its row *exactly*,
where the file's satisfies it only as well as the previous run's Newton
converged. Measured on the `sigma` row, recomputed against loaded —
`LinearDiffusion` 4.3e-19 / 1.5e-18 (both round-off), `AuxVarTest` 6.9e-18 /
5.6e-9, `nonlin` 3.5e-18 / 4.1e-7. No outcome moves: the whole restart
round-trip tolerance matrix is unchanged. And the loaded value is the more
faithful one — it is the `sigma` the previous run was actually integrating with
when it wrote the file, where the rebuilt one is a state that run never had.
The *projection* path still rebuilds it, because a degree change leaves the
stored coefficients in the wrong space.

**`y'` is loaded on a copy-path restart too, and that one is not a saving but
a correctness fix.** `setInitialConditions` builds `dydt` by solving `du/dt`
out of the `u` row and leaving every other component at zero. On a cold start
or a projected restart `IDACalcIC` runs and IDA picks its own first step, so
the zeros are harmless; on the copy path `IDACalcIC` is skipped, and
`IDA_YA_YDP_INIT` would hold the algebraic `y'` fixed anyway. IDA's predictor
then sees sigma, q, lambda, the aux variables and the field standing still
while they move, and the first step's error estimate goes like `h |y'|`
instead of `h^2`: measured on `AuxVarTest`, 4.1e4 at the cadence and falling
only in proportion to `h`, eight error-test failures of the ten IDA allows. So
the copy path takes the file's `dYdt` -- IDA's own derivative when it wrote the
file -- and *then* solves `du/dt` over it as before. Not the file's `du/dt`:
that is an interpolant at the output time and satisfies the `u` rows only to
IDA's tolerance, a weighted residual of 0.3 on the warm `TestDiffusion`
restart against 0.0026 with the solve, which
`a_warm_start_from_a_restart_file_does_not_run_calcic` holds below 0.01.
`psi_round_trips_through_a_restart` compares every other component with the
file bit for bit; with the old zeros, 56 of the 58 entries that are not `u`
differ.

The restart then takes the steps a consistent state allows, which at loose
tolerances can land it further from an uninterrupted run than the old one did
-- the old first step failed down to a tiny `h` and crept forward, accurate by
accident. Compare restarts at tight tolerances: on `LinearDiffusion` at
1e-6 / 1e-8 the restarted run is 1.8e-7 from a 1e-10 reference against the old
2.0e-7, in 36 steps against 48 and with half the error-test failures.
`test_a_run_can_be_restarted_from_its_own_output` runs that way for this
reason.

**`setRestartValues` recovers boundary data only for a case that has none.**
The `LowerBoundary`/`UpperBoundary` defaults read `uL`/`uR`, and a case may
fill those itself -- every `AutodiffTransportSystem` does, from its config --
so recovering them from the restarted profile replaced the case's data with
values sampled off the solution. HDG imposes Neumann and mixed data weakly, so
`q_h` at the face is not the datum: `ADTest`'s zero-flux axis resumed as a
flux of 2.7e-4, the resumed run solved a different problem, and at rtol 1e-6
its first step failed ten times with a `dsm` of 1.17 that no `h` moved -- the
inconsistent-state signature, from a state that was consistent with the
problem it was written by. `uL`/`uR` are now recovered only when absent, or
when an earlier restart recovered them (`lowerFromRestart`), and a Dirichlet
end takes the trace entry, which holds `g_D(t)` exactly.
`a_restart_keeps_the_boundary_data_the_case_owns` is the guard; the same call
is how `DegreeAdaptation` and `MeshAdaptation` hand state between levels, so
those were resampling an autodiff case's boundary data at every rung too.

**A cold start and a projected restart solve for their algebraic `y'`,
exactly, after `IDACalcIC`.** `IDA_YA_YDP_INIT` makes algebraic *values* and
differential *derivatives* consistent and leaves the algebraic derivatives at
zero. On a projected restart -- mid-run, first step at the cadence -- that zero
failed the error test ten times at a cadence of `t0 / 5`; on a cold start it
forced the first step below `MinStepSize`'s default at a tight tolerance, and
10 of the 17 regression configs failed at 1000x their tolerances (all 17 run
now, and 4 at 10x). `consistentAlgebraicDerivatives`
(`Solver.cpp`) solves for them from the requirement that every row of the
residual stays zero along the trajectory, `F_y y' + F_y' u'' + F_t = 0`, with
`u'` given and `u''` and `z'` the unknowns. Where no algebraic row reads `y'`
that is `G_z z' = -G_t - G_u u'` and `u''` is a by-product; where one does --
`ScalarTestLD3`'s controller reads `dM/dt` -- its row reads `u''`, which only
the differentiated differential rows pin down, and setting `u'' = 0` there
gives a `y'` contradicting its own constraint. `J(alpha) = F_y + alpha F_y'`
solves the whole system to O(1/alpha) with `x_d = u' + u''/alpha`:
`J x = alpha F_y' u' - F_t + F_y P_d (x_d - u')`, the last term iterated by
a residual difference -- four or five passes and one Jacobian build on the
fixture. **`alpha` sets the convergence rate, not the answer, and the solve
finds its own.** Each pass contracts by about `s / alpha`, `s` the fastest
rate in the differential rows, so `alpha` has to beat the stiffness. It
starts at 1e8 and, when a pass contracts slower than 1e-3, is multiplied by
the shortfall against 1e-6 and the Jacobian rebuilt, up to a cap: 1e10 where
an algebraic row reads `y'`, since that row's `alpha F_y' x_d` and
`alpha F_y' u'` cancel to its answer with an error of about
`alpha eps |F_y' u'|`, and 1e14 otherwise. It must not be tied to `delta_t`:
that was `1e8 / delta_t`, which on Shestakov's `delta_t = 1e3` stalled and
handed IDA an algebraic `y'` 4-18% out on every mesh.
`the_algebraic_derivative_solve_raises_alpha_until_it_converges` starts it at
1 and requires the default start's `y'`;
`algebraic_derivatives_do_not_depend_on_alpha_where_an_algebraic_row_reads_du_dt`
requires `ScalarTestLD3`'s three starts to agree. `G_t` is the case's own explicit time derivative
(`dSigmaFn_dt`, `dAuxG_dt`, `dLowerBoundary_dt`, ... -- zero by default) through
`explicitTimeDerivative`, which assembles `dF/dt` at fixed state with the
residual's own code on a zero state. **A missing hook is checked for, not
trusted**: a forward difference of the residual in `t` gives `G_t` to ~1e-8
for any case, and where the hooks disagree with it beyond that the difference
is used and the case is warned about. A case that makes its data depend on `t`
and declares nothing otherwise starts from a `y'` contradicting its own
boundary, which is *worse* than zero in the cells at that boundary --
`MatrixDiffusionTest`, missing `dLowerBoundary_dt`, ran at 1000x with zeros and
failed its first step without the check. One residual per pass, and three more. The
restart's counts also survive the `IDAReInit` that hands IDA the derivatives
(`idaResidualEvalsBeforeReinit`), which clears IDA's counters.
`a_cold_start_begins_with_consistent_algebraic_derivatives` is the guard.
