# Steady-state termination on the Newton correction

Date: 2026-10-11
Status: phases 0-3 built on branch `gp-toy-model`, and `NewtonStep` is the
default; phase 4 not started. "As built" records where the implementation
departs from this design and why, and "The migration sweep" what flipping the
default changed. The two measurement drivers cited throughout --
`python-examples/t3d-relu/` and `python-examples/toy-model/noise_study.py` --
are still on that branch; only the solver change came to `main`.

## Why

A steady solve stops when `||F||_resScale < SteadyStateTolerance` (`SteadyState.cpp:623–652`).
That norm is the residual weighted by `1/sqrt(h)` per cell row, with every row
of every channel in one sum. Two measurements say it is the wrong test, one
noiseless and one noisy, both on T3D's ReLU benchmark
(`python-examples/t3d-relu/`: n_e, p_H, p_e at 16 cells, k = 4):

* **It cannot see a channel whose rows are small.** The density's flux vanishes
  at steady state, so its rows hardly enter `||F||`. Noiseless, a solve stopped
  at `||F|| < 3` is out by 2.6% in n, still 1.9% at `||F|| = 0.03`, and below
  1e-3 only at `||F|| = 0.01`. The test fires on the heat-flux rows and passes
  over the density.
* **Under noise it cannot terminate at all.** With +-20% stochastic fluxes
  `||F||` floors at 2.46, set by the heat-flux rows and 250x above the level at
  which n converges. A tolerance above the floor stops with n 2.4% out; below
  it, every step is rejected. Frozen noise moves n by only 3-4e-3, so the error
  is the test's, not the noise's.

The second half of the noisy failure is the **acceptance rule**, MaNTA's own
addition to pseudo-transient continuation (`SteadyState.cpp:653–713`): a step is
kept only if `||F||` fell, and a rejection restores the state and cuts dt by 4.
Under noise `Fprev` ratchets down to the luckiest draw seen so far, after which
almost every step is rejected: dt fell to 8e-60 in a run of ~1000 calls. Kelley
and Keyes take one Newton iteration per step and accept it.

T3D, by contrast, tests the relative change of each profile between iterates
and doubles its step whenever that change is small. Its density converges; on
the same benchmark it is 0.87% out after 68 calls, almost all of it from its
loose 5% threshold rather than from noise.

## What IDA does (confirmed in the SUNDIALS source)

IDA never tests the residual for convergence. It tests the **Newton correction**,
the solution of the linear solve, in its WRMS norm with the error weights
`ewt_i = 1/(rtol |y_i| + atol_i)`:

* **The corrector** (`src/ida/ida_nls.c:347-389`, `idaNlsConvTest`):
  `delnrm = N_VWrmsNorm(del, ewt)`, `del = J^-1 F`. Converged when
  `ss * delnrm <= epsNewt`, where `ss = rate / (1 - rate)`,
  `rate = (delnrm_m / delnrm_0)^(1/m)`, and `epsNewt = epcon = 0.33`
  (`ida.c:183`, `1246`). On the first iteration `ss = 20` (`ida_nls.c:273`) unless
  `delnrm <= 1e-4 toldel` passes outright, and `rate > RATEMAX = 0.9` is a
  recoverable convergence failure.
* **`IDACalcIC`** (`src/ida/ida_ic.c:431-440`, and `IDAfnorm`, `616-660`):
  `fnorm = IDAWrmsNorm(J^-1 F, ewt)` against `epsNewt = epiccon = 0.01 * epcon`.

The `ewt` there is MaNTA's own: `IDAWFtolerances(IDA_mem, getErrorWeights_static)`
(`Solver.cpp:377`). So the test this design wants already runs on every
time-marching step, built from MaNTA's tolerances. `estimateObjective` already
computes the undamped correction at the final state (`SystemSolver.cpp:3986-3993`),
in an unweighted 2-norm.

KINSOL cannot express the same test. Its scaled-step test
(`src/kinsol/kinsol.c:1864`, `KINScSNorm`, `2059`) is a max norm of
`|delta_i| / (1/uscale_i + |u_i|)` with no rate estimate, and returns it as
`KIN_STEP_LT_STPTOL`, "stuck", not as convergence. And it has no hook for a
convergence test of the caller's own.

## The change

### 1. The test

A new key, `SteadyStateTest = "NewtonStep" | "Residual"`.

**`NewtonStep`, the default:** converged when

    ss * || J0^-1 F(y) ||_WRMS(ewt(y)) <= SteadyStateStepTolerance     (default 0.33)

where `J0` is the steady Jacobian (alpha = 0), `F` the steady residual,
`ewt` exactly what `getErrorWeights` gives IDA, and `ss` IDA's rate factor from
successive corrections: 20 on the first, `rate/(1 - rate)` after. The correction
is a dimensionless distance to the solution in units of the user's tolerances,
per component, so the density counts on its own scale.

The rate factor is what makes the test honest with an approximate Jacobian. A
Newton method that converges linearly at rate r is still `r/(1-r)` corrections
from the answer when its correction is small: T3D's measured r = 0.88 gives
`ss = 7.3`, so the test then asks for a correction 7.3 times smaller than a
quadratically convergent one would need.

**`Residual`:** the present test, bit for bit, including the acceptance rule.
It stays for reproducing old runs, and for the cases whose tolerances were tuned
to it.

**Which components.** IDA includes every component unless
`SuppressAlgebraicError` is set, and so does this: the test honours the same key,
masking with `id` as `IDAWrmsNorm(..., suppressalg)` does. Two weights deserve
attention:

* the Dirichlet trace entries, whose correction is identically zero
  (`imposeDirichletTraceRows`), dilute the mean slightly and are harmless;
* `q` and `sigma` share their variable's `atol` (`SystemSolver.cpp:4555-4557`),
  though their units differ. Where a flux vanishes at steady state -- the
  density's here -- its weight is `1/atol`, so its correction must fall below
  `atol` itself. That is what IDA asks too, and why `SuppressAlgebraicError`
  exists; the first measurement below says whether it bites.

### 2. The iteration that supplies the corrections

Recommended: **in `NewtonStep` mode, MaNTA runs the Newton iteration itself**,
mirroring `idaNls`, and KINSOL is kept for `Residual` mode only. The loop runs
under `KIN_NONE` -- no line search -- so KINSOL contributes `y <- y - J^-1 F`, a
Jacobian reuse policy and counters, all of which MaNTA already has
(`residual`, `updateMatricesForJacSolve`, `solveJacEq`). Writing the iteration
puts the correction, its norm and its rate where the test can read them, at no
extra cost: the test uses the correction the iteration computes anyway, as IDA's
does.

The alternative -- keep KINSOL and compute `J0^-1 F` after each accepted step,
as `estimateObjective` does -- costs a Jacobian build and a solve per step, and
leaves the inner solve stopping on `fnormtol`, the residual test this replaces.
Under noise that inner test is what fails, so the alternative does not fix the
noisy case.

**Cost.** A Newton iteration is one residual sweep, plus a Jacobian sweep when
one is built, and the test needs nothing more. The state returned is
`y_{m+1} = y_m - delta_m`, the one IDA returns, which is not evaluated again.
That makes a steady Newton solve `1 + 2n` sweeps against today's `2 + 3n`
(`a_steady_solve_spends_the_physics_sweeps_it_has_to_and_no_others` pins the
latter and moves with it), because there is no separate merit evaluation.

### 3. Pseudo-transient continuation under the new test

* **Accept** a continuation step when its inner iteration converged -- the
  `NewtonStep` test applied to the damped system, at `epsNewt` -- and reject
  only when it did not (`rate > RATEMAX`, or `NewtonMaxIterations` reached).
  No requirement that `||F||` fall. This is Kelley and Keyes, and it is the half
  of the noisy failure the test alone does not fix.
* **Grow dt** by SER on the first correction of each step,
  `dt *= max((d_prev / d_now)^SERRate, SERFloor)`. At the start of a step
  `y = uPrev`, so the damping term vanishes and that correction is
  `(J0 + M/dt)^-1 F(y)` -- the steady residual through the damped operator.
  Under noise the ratio hovers near 1 and the floor (2 by default) keeps dt
  doubling, which is the behaviour T3D shows and the reason its slow channel
  relaxes.
* **Declare convergence only on an undamped correction.** At finite dt the
  correction is not the steady one, and a small enough dt makes it small. Two
  ways to get one: switch to Newton when dt passes `PseudoTransientMaxStep`
  (or a new threshold), or form `J0^-1 F` at the end of an accepted step. The
  second costs a Jacobian assembly at alpha = 0 but not a physics sweep, if the
  derivatives taken at that state are cached for the next build; that cache is
  worth having anyway. Recommended: the switch, measured first, since it costs
  nothing.

### 4. Configuration

| Key | Type | Default | Meaning |
|---|---|---|---|
| `SteadyStateTest` | string | `"NewtonStep"` | which test, and which acceptance rule |
| `SteadyStateStepTolerance` | double | 0.33 | `epsNewt` for the steady test (IDA's `epcon`) |
| `SteadyStateTolerance` | double | 1e-3 | unchanged; read only under `Residual` |

`SteadyStateTolerance` keeps its other role -- its *presence* arms steady-state
termination on both surfaces -- so no config stops terminating. Under
`NewtonStep` its value is unused, and the schema should say so in
`--list-options` rather than leave it silently ignored.

The default changes answers. Every steady config in the tree is then converged
to its `Relative_tolerance` and `Absolute_tolerance` rather than to a residual
threshold, and several set those for IDA's benefit rather than the steady
solve's (`python-examples/toy-model/run.conf` has `rtol = 1e-3`). Migration
is a sweep, not a judgement per file: run every steady case under both tests
and record where the answers differ by more than the regression tolerance.

### 5. Noise: phase 2

The test above fixes the noisy failure's *cause* -- the density now counts -- but
not its *floor*. With fresh noise on every evaluation the correction stops
falling at `|| J0^-1 eta ||`, `eta` the noise in F, and the rate estimate goes to
1. IDA would call that `rate > RATEMAX` and cut h. A steady solve needs a
different answer:

* **A `NoiseLimited` outcome.** When the correction has stopped falling -- rate
  above `RATEMAX` over a window of iterations, with the norm bounded -- the solve
  stops, returns `SteadyOutcome::NoiseLimited` and reports the floor it reached.
  A sliced driver can then decide what to do.
* **Iterate averaging.** Near the solution with a good Jacobian, each noisy
  Newton iterate is `y* - J0^-1 eta_m`, with `eta_m` independent per
  evaluation, so the mean of M iterates is M^(1/2) closer to `y*` for M more
  calls. That is the cheapest accuracy available under stochastic fluxes, and
  neither T3D nor the present loop takes it. Returned on `NoiseLimited`, behind
  a key, measured against T3D's single last iterate.
* **A noise level from the case.** A flux model like GX knows the standard error
  of its own time average. An optional hook returning it per point turns the
  floor into a number, `|| J0^-1 sigma_F ||_ewt`, and the test into
  `ss * delnrm <= max(epsNewt, c * floor)`: stop when the correction is
  indistinguishable from noise. Without the hook, two evaluations at one state
  estimate it for one extra call.

Each of these is measured on `python-examples/toy-model/noise_study.py` and
`python-examples/t3d-relu/noise_benchmark.py` before it is kept.

## As built (2026-10-12)

**Phase 0 held.** `SteadyStateMeasureCorrection` reports `||J0^-1 F||_ewt` at the
end of a `Residual` solve. On the T3D benchmark at rtol = atol = 1e-8, the
noiseless `||F|| < 3` stop -- n out by 2.6% -- measures 2.2e6 tolerance units,
i.e. 2.2e-2 relative: the quantity sees exactly the error `||F||` missed.

**The inner solver is `SUNNonlinSol_Newton`**, SUNDIALS' generic Newton, with this
test as its `ConvTestFn` -- the hook IDA itself uses -- rather than a loop of
MaNTA's own. KINSOL stays for `Residual`. `2026-10-12-kinsol-convergence-test-hook-design.md`
is the upstream KINSOL change that would let `NewtonStep` return to KINSOL.
Jacobian reuse within a solve (`NewtonJacobianReuse`) is done in the `LSolve`
callback, since `SUNNonlinSol_Newton` sets up only at the start of a solve.

Departures from IDA's test, each measured on the T3D benchmark:

* **No divergence test on a continuation step.** IDA's `rate > RATEMAX = 0.9`
  rejected the first four steps at dt = 1e-2 at their second correction and cut
  dt to 4e-5, and so did CVODE's gentler `delta > 2 delta_prev`; KINSOL, which
  has none, converged those steps in 8-10 iterations. A continuation step is
  capped by `NewtonMaxIterations` alone, and a non-finite correction fails it.
  `RATEMAX` is kept for the undamped probes, which should fail fast.
* **`ss` is never below 1**: the distance left is taken to be at least the last
  correction, as in Deuflhard's error-oriented Newton termination. IDA trusts a
  rate from one ratio because its predictor starts it near the answer; a cold
  steady solve need not be in Newton's asymptotic regime after two iterations.
  Found by the migration sweep: on `two-temperature-critical-gradient`
  corrections of 1883 then 22.6 tolerance units gave rate 0.012 and
  `ss * |d| = 0.27`, with T_e still 13 tolerance units (RMS) from the converged
  state. With the floor it takes 5 iterations and lands on it.
* **The rate is `max(IDA's, the last ratio)`.** IDA's geometric mean since the
  first correction suits its four iterations; over twenty, a fast first drop
  and a slow linear tail make it underestimate the current rate. The guard took
  p_H's error at the stop from 10% to 4.5% at rtol 1e-2 (exact derivatives) and
  p_e's from 6.5% to 3.3% (T3D's Jacobian), for at most 4 more box calls.

Departures from section 3:

* **SER reads the steady residual, not the first correction.** The correction
  `(J0 + M/dt)^-1 F` grows with dt for a fixed F, so its ratio across a dt change
  is not a progress measure. The steady residual is free: the first evaluation
  of every solve is at ycor = 0, where the damping vanishes. It is known only at
  the start of a step, so the growth lags one step.
* **The undamped solve is a probe, attempted from a damped step**, not a switch
  at a dt threshold, which would need a dimensional key. A probe is made once a
  damped step's first correction is within one tolerance unit (or dt is at
  `PseudoTransientMaxStep`); a failed one costs two residuals and two builds,
  and the next waits for dt to grow sixteenfold.

**`SuppressAlgebraicError` is required under noise.** sigma is an algebraic
unknown equal to the flux, so a noisy flux puts its noise into sigma's
correction at full strength, undamped by dt: with sigma in the norm no damped
step converged and dt fell to 1e-14. With it masked -- what IDA's key does --
the noisy benchmark converges. This answers the q/sigma question below for the
noisy case.

**Measured** (`python-examples/t3d-relu/README.md` has the tables). Noiseless,
NewtonStep converges what `Residual` cannot see: 1.2e-3 in n at rtol 3e-3 in 33
box calls, against 2.6% at `||F|| < 3`. With +-20% noise at rtol 1e-2 it
converges seven seeds of eight in a median 11 calls with exact derivatives, and
eight of eight in 65 with T3D's Jacobian, against T3D's own 68. At rtol 3e-3
with exact derivatives no seed converges: the test asks for less than the noise
floor, which is phase 4. (The `ss >= 1` floor cost one seed at 1e-2 and two at
3e-3 with T3D's Jacobian; noiseless correctness comes first.) In its own norm
the test is conservative with exact derivatives and up to 5x optimistic with
T3D's Jacobian; max-norm errors reach 2.4 tolerance units, as an RMS test
allows.

### Kelley-Keyes continuation (2026-10-12, later)

The first build kept a damped step whenever its inner solve converged on the
same test. That costs at least two iterations per damped step -- the rate needs
a second correction -- and made `PseudoTransient` dearer than under `Residual`
(Park 11 visits against 6). Worse, a step whose inner solve ran out of
iterations was rejected with its progress discarded: on Shestakov's graded mesh
the warm start resumed at dt = 3e8, where Newton contracts only linearly (2/3
per iteration, the triple root of `q^3` at the axis), and nineteen 20-iteration
rejections cost 23x the uniform solve.

So a damped step is now **one Newton iteration, accepted as it stands** --
Kelley and Keyes's pseudo-transient continuation proper -- and only the undamped
solve is iterated and tested. Two things make that safe:

* **SER shrinks dt on a step that raised the residual** (by the same ratio); the
  floor applies only to a step that reduced it.
* **A correction that reverses the last one halves the step** (cosine below
  -0.5 in the error weights; two steps without a reversal double it back
  towards 1). Single iterations cycle either side of a kink in the flux --
  `chi ~ (|q| - q_c)^(1/2)` -- and no dt stops them, because the kink is in
  sigma and q, which are algebraic and undamped. On the two-temperature
  problem 0.0054 above its threshold, plain single iterations alternated
  residuals 0.34 and 1.76 for 200 steps at every dt from 3.7e5 down to 1.
  Recovering after one quiet step instead of two let a period-four cycle
  replace the period-two one.

Rejected on the way, each measured on that case and on Shestakov: holding dt
after a rise (still cycled); undoing a step whose successor starts more than
twice as high and retrying it (cycled at every retry dt, since the overshoot is
dt-independent); and solving the retry to convergence (converged at 747 visits,
failed from the constant-chi start, and put Shestakov's graded mesh back at
36500 evaluations).

Measured, visits per point, before -> after: Park continuation 11 -> 7 (`Residual`
6); Jardin continuation at reuse 10, 110 -> 14; Shestakov continuation at reuse
10, 346 -> 30; the near-threshold two-temperature case 229 / 269 -> 33 / 73
(continuation / Newton; time marching 209); Shestakov's graded mesh 42500 ->
5100 evaluations (`Residual` 6000). Every answer stays within 5e-3 of the
converged discrete solution. Under noise the plain version took the ReLU
benchmark at rtol 3e-3 with exact derivatives from 0/8 seeds to 7/8; the
reversal halving, which noise near the floor trips at random, gives most of that
back (2/8), and telling noise from a kink is left to phase 4.

## The migration sweep (2026-10-12)

Every steady configuration in the tree, run with the default flipped and with
`Residual` named, compared as the regression suite compares -- relative L2 of
each variable's u, against 5e-3:

* the five `python-examples/*/run.conf` through `manta.cli`;
* every `python-examples` driver that reaches `run_ss`, with each solve's final
  state captured and matched in order (a `sitecustomize` hook injecting the test
  into every `Runner.configure` that named none): 17 drivers, 576 captured solves,
  the FFI ones on an `XLA_FFI` build;
* the unit and Python suites, which take the default wherever they do not name
  a test.

Results, after the `ss` floor above:

* **Every compared solve agrees within 5e-3**, almost all to 1e-7 or better.
  The largest, 3.9e-3 (Jardin in `paper-figures/steady_modes.py`), is the
  `Residual` side stopping short at its loose `SteadyStateTolerance = 0.01`:
  against the exact solution `Residual` is 1.7e-3 out and NewtonStep 1.5e-8.
* **Cost** falls on the CLI configs (Park 10 residuals to 5, Jardin 13 to 8,
  two-temperature 8 to 5, thermodiffusive 6 to 3) and on most drivers
  (`reference-algorithms` 524 to 278, two-temperature's benchmark 1158 to 354),
  and rises a little where continuation dominates (`adaptivity.py` 1985 to
  2212). At 1e-11 the mode comparison in `docs/running.rst` was re-measured
  under both tests; NewtonStep is cheaper or level on all six continuation
  entries.
* **Three drivers stall** -- `nc-toy-model/nc.py`, `nc-parallel/nc_parallel.py`,
  `nc-parallel/nc_multigpu.py`: a graded wall layer (cells ~5e-4) puts the
  round-off floor of the correction at IDA's threshold for atol 1e-6, so
  continuation runs out of steps with `||F||` at 1e-9. They now name
  `Residual`; atol 1e-5 also converges, and is the alternative.
* **Pinned to `Residual` without a sweep**: the stellarator scripts
  (`python-physics/stellarator`, an `XLA_FFI` build, DESC and yancc, and marked
  unverified), whose `SteadyStateTolerance` of 1e-5 sits beside
  `Relative_tolerance = 1e-2`, so the default would loosen them to 1%; and
  `toy-model/noise_study.py`, which measures the residual test itself.
  `t3d-relu/noise_benchmark.py` names its test per run already; the mirror
  plasma never takes the steady path.
* **Tests**: ten C++ tests and two Python ones exercise the KINSOL path's
  mechanics (its sweep budget, counters, the ||F|| rejection rule, a tolerance
  nothing reaches) and now name `Residual`. One is a real difference rather
  than mechanics: `NewtonMaxIterations = 1` makes a Residual continuation work
  harder, but leaves NewtonStep no damped step that converges.

Found on the way, both pre-existing and independent of the test:
`PyRunner::getSolution` segfaults for a variable index past the last, and after
a steady solve that failed under `MeshAdaptation`.


## Phases

0. **Measure.** *Done.* Report `ss * ||J0^-1 F||_ewt` in the steady stats at the end of
   every solve -- one build and solve, or none where `estimateObjective` has
   already formed the correction, which then shares it. On the T3D benchmark
   this must read large at every noisy stopping point and at the noiseless
   `||F|| < 3` stop, and small at the converged solution. If it does not, this
   design is wrong and stops here.
1. *Done.* **`NewtonStep` in Newton mode**: MaNTA's own corrector loop, the test, the
   config keys, `Residual` kept bit for bit. Default still `Residual`.
2. *Done.* **Pseudo-transient continuation** under the new test: acceptance, SER on the
   correction, the switch to an undamped correction.
3. *Done.* **Flip the default**, with the migration sweep in the same change.
4. **Noise**: `NoiseLimited`, averaging, the noise hook.

## Tests

* `Residual` mode is bit for bit: every steady unit test and regression output
  unchanged, `cmp` over the steady configs' `.nc`.
* The test fires at IDA's threshold: a case with a known Newton contraction,
  converged until `ss * delnrm` crosses 0.33, checked on both sides.
* The density case: a three-channel fixture with one vanishing-flux channel,
  noiseless; `Residual` at a loose tolerance leaves it unconverged and
  `NewtonStep` does not. This is the failure in Why, pinned.
* The sweep budget `1 + 2n` in Newton mode.
* Acceptance under a deterministic perturbation of F (seeded, so reproducible):
  no step rejected for a residual rise alone.
* A sliced `NewtonStep` solve resumes without re-climbing, as
  `a_resumed_steady_solve_does_not_re_climb_the_ser_ramp` requires today.

## Open questions

* **One threshold or IDA's two.** IDA uses 0.33 for the corrector and
  `0.01 * 0.33` for `IDACalcIC`, because a corrector feeds a local error test and
  an initial condition feeds nothing. A steady answer feeds nothing either,
  which argues for the tighter one. 0.33 matches the corrector you named, so it
  is the proposed default, but this is the choice to make deliberately.
* **The `q`/`sigma` tolerance.** Sharing `u`'s `atol` was harmless while only
  IDA's error test read it. A steady test that converges *to* those weights may
  want per-row-kind tolerances; phase 0 says whether it does.
* **The adjoint.** It runs at the steady state the solve returns, so it inherits
  whatever the test accepts. Nothing changes structurally, but gradients are
  now taken at a state converged in `ewt` rather than in `||F||`.
