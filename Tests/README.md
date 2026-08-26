# MaNTA test suites

Three suites, all registered with CTest (`ctest --test-dir build`) and all
runnable from any working directory.

| Test | Suite | Location |
|---|---|---|
| `unit` | Boost.Test C++ unit tests | `Tests/UnitTests/` |
| `regression` | Solver run against checked-in `.ref.nc` references | `Tests/RegressionTests/` |
| `python` | pytest suite for the pybind11 module | `python/Tests/` |

Each also has a build target of the same name plus `_tests` — `unit_tests`,
`regression_tests`, `python_tests` — which builds what it needs and runs that one
suite.

The Python suite writes its solver output into the *current directory*, not into
pytest's `tmp_path`: `OutputFilename` is passed to `setInputFile`, and
`Solver.cpp` keeps only `inputFilePath.stem()`. Tests that need output files
therefore use unique cwd-relative names and clean up after themselves; see
`test_output_filename_keeps_only_the_basename` in `test_runner.py`, which pins
the behaviour.

A run writes netCDF (`<stem>.nc`, `<stem>.restart.nc`) by default and no text
output at all; `.dat` files need `WriteDatFile` (and `WriteDebugDatFiles` for
the `.dydt.dat` / `.res.dat` pair), both off by default. Test cleanup code must
therefore treat `.dat` as optional.

`cmake --preset coverage && cmake --build build-coverage --target coverage`
builds with `--coverage -O0`, runs all three, and writes
`build-coverage/coverage/index.html` (numerical core + Python bindings) and
`.../physics.html` (`PhysicsCases/`, informational). See the README.

Note that the unit tests now run from the **build directory** rather than the
repo root, because CTest launches them there. Fixtures are unaffected — they are
reached through the absolute `TEST_DATA_DIR` — but a test that writes output and
cleans up after itself is doing so in `build/` now.

## Unit tests

Boost.Test in header-only mode (`boost/test/included/unit_test.hpp`), so there
is no `-lboost_unit_test_framework` link step -- `libboost-dev` is enough.
`Tests/UnitTests/main.cpp` defines the module; every other `.cpp` listed in
`MANTA_TEST_SOURCES` (`Tests/UnitTests/CMakeLists.txt`) contributes suites.

Two things worth knowing when adding a test:

* **Private access.** Tests exercise the HDG block assembly, the static
  condensation solve and the adjoint vectors directly, all of which are private
  to `SystemSolver`. A `-DTEST` build widens that section via
  `MANTA_TEST_PRIVATE` (`SystemSolver.hpp`), so no per-test friend declaration
  is needed. This replaced a scheme that befriended one generated struct per
  `BOOST_AUTO_TEST_CASE` and needed a new forward declaration in the header for
  every new test.
* **Fixture paths.** netCDF fixtures (`testic.nc`, `MatrixDiffusion.restart.nc`,
  `Bfield.ref.nc`) must be reached with `testDataPath()` from `TestPaths.hpp`,
  which resolves against the `TEST_DATA_DIR` baked in by
  `Tests/UnitTests/CMakeLists.txt`. Do not hardcode `./Tests/UnitTests/...`; that
  only works from the repo root, which is no longer where the binary runs.
* **A passing run is silent.** Several tests deliberately provoke output --
  they run the full solver, hand `ErrorChecker` a null pointer, or make the
  physics throw so `static_residual` has to report it. Wrap those calls in a
  `CapturedOutput` (`CapturedOutput.hpp`) so the noise does not bury a real
  failure. It redirects at the file-descriptor level. The project's own output
  is all `std::print`/`std::println` now, but that still lands in two different
  places -- `std::print(stderr, ...)` writes to the C `FILE*` while
  `std::print(ofstream, ...)` goes through the stream -- and SUNDIALS' error
  handler writes to stderr from C regardless. Only the descriptor is common to
  all three, so swapping `std::cout`'s streambuf would not do.

  Two rules when using it. **Capture the noisy call, restore, then assert** --
  Boost.Test writes failures to stdout, so an assertion that fires while
  captured is swallowed. And prefer asserting on `capture.text()` to merely
  discarding it: `ErrorChecker` and `logmsg` exist to say something useful, so
  checking *what* they printed is worth more than checking that they did not
  throw.

Run a single suite or case directly:

```sh
Tests/UnitTests/UnitTests --run_test=dg_approx_tests/grid_test --log_level=all
```

## Regression tests

`TestSolutions.py` runs the solver over each `.conf` and compares in four modes:
against a closed-form analytic solution, against a steady state, against a
checked-in `<name>.ref.nc`, and -- for three cases -- against itself via a
restart round trip. Tolerance is 5e-3, overridable with `--tolerance`.

**It compares only `u`.** `test_ref_soln_l2` reads
`groups[name].variables["u"]`, so the `q`, `sigma` and `u_star` that every run
also writes are never checked against a reference. That is not hypothetical
slack: making t0 output report the CalcIC-corrected state moved the t0 `sigma`
of `AuxVarTest` by 2.3e-2 and the t0 `q` of `nonlin` by 1.2e-3, and the suite
could not see either -- the change is invisible in `u` precisely because `u` is
differential and `IDACalcIC` holds it fixed, so the fields that move are the
ones nothing looks at. Closing it means regenerating the references, which hold
the pre-CalcIC t0 values, and picking tolerances for three fields that are a
derivative rougher than `u`; it is a piece of work rather than a line in the
harness, which is why it is in `TODO` and recorded here rather than done.

The restart round trip runs to `t_final` in one go, then again in two halves via
a `.restart.nc` file, and requires the final states to match. It is the only
test of `WriteRestartFile` -> `StoreGridInfo` -> the restart branch of
`runManta` -> `setRestartValues`, and it is what would catch a repeat of the
clustered-grid contiguity defect (a grid rebuilt from a restart file must
compare *equal* to the one that wrote it).

**All three round trips survive 1e-6 / 1e-8** -- see the measured table beside
the calls in `TestSolutions.py`.

The ceiling above that belongs to the *cases*, not to the restart path. At
1e-8 / 1e-10 `MatTest`'s **uninterrupted** run fails as well, so nothing there
implicates restarting; the other two resume-fail with `IDA_ERR_FAIL` (-3).
`LinearDiffusion` and `AuxVarTest` run at 1e-6; `MatTest` stays at 1e-4 because
1e-6 costs 101 s against 6.0 s for agreement of 2.7e-10 that nothing needs.

Two things are load-bearing for that, and both are easy to undo by accident:
`setInitialConditions` must keep the trace a restart file carries rather than
rebuilding it with `EvaluateLambda()`, and `AuxVarTest` must declare
`dSigma_dPhi` for *both* variables (both described below).

`.ref.nc` files are committed -- `.gitignore` has `*.nc` with a `!*.ref.nc`
negation.

**`AuxVarTest.ref.nc` has been regenerated twice, both times because the case was
running too loose to pin a solution at all.** That is the thing to understand
before touching it again: a reference is only a reference if the run that made it
is more accurate than the tolerance it is compared at, and for years this one was
not.

The first regeneration followed the `dAux_Mat` column-layout fix (see the
paired-implementations section below); the old reference came from `f35e3ee`, the
commit that introduced that defect, so it recorded a run whose Jacobian was
missing `dG/du` entirely.

The second followed a *second* missing Jacobian block in the same case.
`SigmaFn` adds `(a - u*u)` to **both** variables' fluxes, but `dSigma_dPhi`
declared the derivative only for variable 0. On the constraint manifold
`a = u*u`, so the stray term vanishes and nothing notices -- and a warm start is
precisely the state that is *off* the manifold, where Newton then diverged.
Finite-differencing `residual` against the assembled Jacobian put the `sigma[1]`
block out by 98% of the residual scale before the fix and 4.2e-9 after it, every
other block unchanged. The fix declares the derivative rather than removing the
term from `SigmaFn`, so the physics -- and the solution -- are untouched.

Both times the trigger was the same and was not the fix: at
`Relative_tolerance = Absolute_tolerance = 1e-2` **the case's own answer is 4.1%
from the converged one**, against a comparison tolerance of 5e-3. Measured
against a rtol = 1e-10 run of the same discretisation:

| run | max relative L2 from converged |
|---|---|
| the 1e-2 configuration | 4.1e-2 |
| 1e-4 / 1e-6 | 3.5e-5 |
| **1e-6 / 1e-8 (now)** | **4.2e-7** |

So any change that moved the Newton path moved the output past the threshold
without either answer being wrong -- the two 1e-2 answers straddle the converged
one, each 1-4% out. The reference was pinning a step sequence.

`AuxVarTest.conf` now runs at `1e-6 / 1e-8`, four orders inside the comparison,
and the reference is a solution. What that buys, measured by forcing different
step sequences on the tightened configuration (`initialTimestep`, `MinStepSize`
within its working range, `AggressiveTimesteps`): the answer moves by **at most
1.8e-6**, where the same class of change used to move it by 0.93% and fail.

Two costs, neither of them free:

* **`MinStepSize` is load-bearing and must stay in the config.** At the default
  1e-7 the case dies with `IDA_ERR_FAIL` (-3) at these tolerances. The limit is
  between 1e-8 (fails) and 1e-9 (works); the config sets 1e-12.
* **An explicit `initialTimestep` above ~1e-4 now fails** where 1e-3 worked at
  1e-2. Nothing in the tree sets it for this case -- the default of 0 lets IDA
  choose -- but a first step too large for the requested accuracy is not
  recoverable, which is the same family as the `MinStepSize` floor.

Runtime went from 0.07 s to 0.15 s.

Earlier references are kept out of the tree; recover one from
`git show <commit>:Tests/RegressionTests/AuxVarTest.ref.nc` if a comparison ever
needs repeating.

## Reading the coverage number

One caveat before chasing the headline percentage: **gcov counts a templated line
once per instantiation**, and `NetCDFIO.hpp` is almost entirely templates
(`AddVariable`, `AppendToVariable`, `AppendToGroup`) instantiated from several
translation units.

That one 128-line header is reported as **625 lines**, of which only 18% are
covered -- yet just **39 distinct source lines** are actually uncovered. It
contributes roughly 40% of the whole uncovered total on its own and depresses
the in-scope figure by about 8 points:

```
headline (as gcovr reports it)                    3726/4920 = 75.7%
NetCDFIO.hpp counted once per source line         3698/4423 = 83.6%
```

`util/trapezoid.hpp` (75%, 17 distinct uncovered lines out of 191 reported),
`Logging.hpp` (75%, 6 distinct) and `DGApprox.hpp` (86%, 10 distinct) are
distorted the same way, less severely.

**Writing tests for it makes the reported denominator grow, not shrink.** Adding
`NetCDFIOTests.cpp` took the header from 551 reported lines to 625, because each
distinct callable passed to `AddVariable`/`AppendToVariable` is a new template
instantiation with its own line records. The distinct uncovered lines really did
improve (48 -> 39) but the percentage only moved from 7% to 19%. The metric is
structurally unable to reflect progress on this file.

`gcovr --merge-mode-functions` does not help either -- it merges functions, not
per-instantiation line records (all three modes give identical output).

So: treat the headline as a floor, and judge work on this header by the count of
*distinct* uncovered lines, not by its percentage.

**And a number is only worth reading if the Python suite ran against the
instrumented module.** The extension lives at `python/manta/_manta<abi>.so` --
in the source tree, because that is where `import manta` has to find it -- so
every build directory writes to the same path, and a run once imported the
Release module while believing otherwise: 133s against 748s for the same tests,
with the report still looking right because gcov data accumulates. Each build
directory now claims the module and replaces one it does not recognise, and the
`coverage` target refuses to start unless what is in place carries
instrumentation; `python/CMakeLists.txt` has the full account. Nothing is needed
from you, but if a binding-layer figure ever looks impossibly low, that is the
first thing to suspect.

## The scalar (Woodbury) path in solveJacEq

`SolveJacTests.cpp` builds the Jacobian by finite-differencing `residual` under
IDA's convention (`J = dF/dY + cj dF/dY'`) and requires the vector the linear
solve returns to satisfy `J dy = g`. With `nScalars = 0` that passes at 3e-10.
With `nScalars = 3` (`ScalarTestLD3`) it came out **O(1)** -- and not only in the
scalar rows, so it did not say whether the fault was in `solveJacEq`'s bordered
elimination or in the physics case's hand-written derivatives.

`ScalarJacobianTests.cpp` settles it by supplying scalar systems whose Jacobians
are known in closed form:

    d_t u = d_x( kappa d_x u ) + COUPLING * mu        G = mu - BETA * Int u dx

so `v = COUPLING`, `w = -BETA * Int phi`, `N = 1`, with a differential variant
(`G = d(mu)/dt - BETA * Int u dx`, giving `N = alpha`) run at two values of
alpha, because with an algebraic scalar the `alpha * dG/dmu'` term is identically
zero and any handling of it looks correct.

**`solveJacEq` is correct**: both satisfy `J dy = g` at 1e-10. The O(1) failure
was three separate defects, none of them in the elimination:

1. `ScalarTestLD3::ScalarGPrimeExtended` reported `dG_0/du = -Int phi` where
   `G_0 = E - (M0 - M)` gives `+Int phi`. `w` enters the bordered elimination, so
   one sign there corrupts the *whole* solve -- which is what made the original
   symptom ambiguous.
2. `ScalarTestLD3::dSources_dScalars` assigned `v[0]` and `v[1]` and left `v[2]`
   alone, and `dSources_dScalars_Mat` hands it an uninitialised Eigen vector.
   Undefined behaviour, and a garbage column in the scalar coupling matrix.
3. `dSources_dScalars_Mat` integrated `dS/dmu` exactly by quadrature, while
   `residual` uses `InterpolateOntoBasis( I, S(nodes) )` -- the projection of the
   *interpolant* of S. The two agree only when `dS/dmu` is a polynomial the basis
   represents; `ScalarTestLD3`'s is a narrow Gaussian, and they differed by 7% of
   the residual at k = 2 on 4 cells. Now interpolatory, like every other block in
   `Matrices.cpp`.

Two smaller things fell out: `dG_1/dI = -gamma_I` was never differentiated
(latent, `gamma_I` defaults to 0), and the case integrated its own mass with a
*global* adaptive Kronrod rule over a piecewise polynomial, so the integral was
not a smooth function of the coefficients -- the finite-difference reference
disagreed with the exact `Int phi` by 8% at k = 4 on 16 cells. Both fixed;
`ScalarTestLD3` now agrees at 1e-9 or better across k = 2..6 and 4..32 cells.

The reusable piece is `checkScalarDerivative`: finite-difference a case's own
`ScalarGExtended` and require `ScalarGPrimeExtended` to match, coefficient by
coefficient. That answers "does this physics case report its own scalar Jacobian
correctly" for any case, and is the first thing to run when a scalar system
converges slowly.

## Order of accuracy

`Tests/UnitTests/MMSConvergenceTests.cpp` is the strongest correctness signal in
the suite: it exercises the residual, the block assembly, the static-condensation
solve and IDA's time integration *together*, and measures the observed L2 order.
A wrong sign in a single block still converges -- to the wrong answer -- but not
at the right rate to the right limit.

Measured, for `u(x,t) = sin(pi x)(1 + t)` on `[0,1]`:

| k | grids | observed order | expected |
|---|---|---|---|
| 1 | 4, 8, 16, 32 | 1.96 | 2 |
| 2 | 4, 8, 16 | 2.97 | 3 |
| 3 | 4, 8, 16 | 3.98 | 4 |

Two things make this work, and both are easy to get wrong when adding a case:

* **The manufactured solution must satisfy the boundary conditions the physics
  case imposes.** `sin(pi x)` vanishes at both ends for every `t`. This is what
  the removed `UseMMS` option on `LinearDiffusion` got wrong: its manufactured
  solution was the initial Gaussian, about 0.29 at the domain edge against a
  boundary condition of 0, so an order study against it could not show `k+1`.
  `LinearDiffSourceTest`'s was worse -- it read `useMMS` and never applied
  `MMS_Source`, so the option silently did nothing. Both are gone; the
  manufactured problems here are self-contained and never depended on them.
  `AutodiffTransportSystem::MMS_Solution` is still there, but nothing overrides
  it any more: the C++ `MirrorPlasma`, its only user, has been removed in favour
  of `python-physics/mirror-plasma`.
* **The time-integration tolerance must be well below the spatial error, but not
  so tight that IDA cannot start.** At 1e-12 it fails at `t = 0` for `k >= 2`
  with "the error test failed repeatedly or with |h| = hmin"; 1e-9 leaves three
  orders of margin over the smallest spatial error in the sweep.

## Order of accuracy with a field model coupled

`Tests/UnitTests/MMSFieldTests.cpp` is the same instrument pointed at the
self-consistent field coupling, and it is the only test class that can catch an
error in the coupled *equations*. The split is worth restating, because the two
halves catch disjoint things:

* The coupled Jacobian is never assembled, so a wrong `A1` or `A2` costs Newton
  iterations and nothing else. Only `FieldJacobianTests.cpp` sees that.
* A sign error in the coupled *residual* converges at the right rate to the
  wrong function. Only a closed-form comparison sees that -- which is this file.

The problem is `u_t - d_x[ g(x; psi) kappa u_x ] = S` with `kappa = 0.7`,
`u = sin(pi x)(1 + t)` and `g` supplied by the field model, all in `MMSHarness.hpp`
alongside the uncoupled sweeps. `kappa != 1` deliberately: a case that used the
geometry slot *as* the diffusivity rather than as a factor multiplying it would
be indistinguishable at `kappa = 1`.

**The manufactured source is checked before the solver is.** `S` is derived
symbolically as `u_t - kappa (g' u_x + g u_xx)`, and
`the_manufactured_source_is_consistent_with_the_exact_solution` evaluates
`u_t - d_x[kappa g u_x] - S` at 21 and 30 `(x, t)` points respectively. Both
models come back at **5e-13**, against a 1e-10 threshold. The derivative there is
a six-point `O(h^6)` stencil at `h = 2e-3`, not a plain central difference: the
flux carries `cos^2(pi x)`, so its seventh derivative goes as `(2 pi)^7` and the
obvious `h = 5e-3` lands at 4e-11 -- close enough to the threshold to be worth
not doing.

Measured local orders, `t = 0.25`, grids 4, 8, 16, 32, `Superconvergent = false`:

| model | k | u | u* | psi |
|---|---|---|---|---|
| `ManufacturedField` (1 DOF) | 1 | 1.90, 1.96, 1.98 | 2.21, 2.09, 2.05 | 2.09, 2.07, 2.04 |
| `ManufacturedField` (1 DOF) | 2 | 2.95, 2.98, 2.99 | 4.47, 4.08, 3.96 | 4.82, 3.93, 3.87 |
| `ManufacturedField` (1 DOF) | 3 | 3.96, 3.99, 4.00 | 4.99, 4.89, 4.71 | 5.12, 4.37, 4.10 |
| `ManufacturedFieldVector` (5 DOF) | 2 | 2.93, 2.98, 2.99 | 4.04, 4.01, 4.00 | 4.03, 4.01, 4.00 |

and `u*` with the flag off against the flag on, `ManufacturedField`:

| k | `u` flag on | `u*` flag off | `u*` flag on |
|---|---|---|---|
| 1 | 1.87, 1.96, 1.98 | 2.21, 2.09, 2.05 | **3.12, 3.05, 3.01** |
| 2 | 2.92, 2.97, 2.99 | 4.47, 4.08, 3.96 | 4.24, 3.99, 3.97 |
| 3 | 3.94, 3.98, 3.99 | 4.99, 4.89, 4.71 | 5.27, 4.98, 4.94 |

Three things to read out of that.

* `u_h` holds `k+1` at every degree and on both models, which is the headline.
* **`u*` reaches `k+2` with the flag on at every degree, so the fourth test
  asserts it** rather than asserting that the flag throws. Geometry is a function
  of `(psi, x)` and the star nodes are just more `x`, so the coupling needs no
  special case in `ComputePhysics`'s `states.size()` loop -- and does not get one.

  **`k = 1` is the row that earns the test**: it is the only configuration in the
  file where the flag-on assertion is not also satisfied flag-off, i.e. the only
  one showing the flag *doing* something rather than failing to break something,
  and it is asserted as such. That is not a new phenomenon -- it reproduces,
  under coupling, the `k = 1` / `k = 2` split the uncoupled study measures and
  the next section records as unexplained: flag off, the interpolatory scheme's
  postprocessing superconverges at `k = 2` but not at `k = 1`. The coupling
  neither causes nor cures it. The `k = 3` flag-off row is the one genuinely new
  number, and it *decays* -- 4.99, 4.89, 4.71 -- the same shape as the nonlinear
  flux's transient superconvergence below, where the flag-on column does not.
* **`psi` starts above `k+1` and slows down**, which is why the assertion is at
  `k+1` and why the extra order is not claimed. `k = 2` gives 4.82, 3.93, 3.87
  and `k = 3` gives 5.12, 4.37, 4.10. There *is* a mechanism for `k+2` -- the
  field quadrature is exact on a degree-`k` field, so `psi_h` is exactly
  `Int u_h dx` and its error is `Int (u_h - u) dx`, a linear functional of the
  error rather than its `L2` norm, which superconverges by the usual duality
  argument -- but a falling rate at `n = 32` is precisely the pattern this
  codebase has already measured and been caught by. See "the two italicised
  flag-off entries" below: the nonlinear flux's `u*` fell by 6.9, 11.7, 9.1 and
  then 2.3, so a sweep ending at `n = 32` reported 3.21 and looked perfectly
  healthy. Until this sweep is refined far enough to tell a settled `k+2` from a
  pre-asymptotic transient, `k+1` is what the evidence supports. The multi-DOF
  model's `psi` is the one column that does *not* decay (4.03, 4.01, 4.00), and
  it is asserted at `k+1` too, on the same three-refinements-is-not-settled
  grounds.

### Which solve produced these numbers

`solveCoupledJacIterative` escalates to the exact Schur solve when it exhausts
`FieldSolveMaxSweeps`, so a sweep that never converged would yield exactly the
exact path's answer with nothing in the result to say so.
`the_coupled_problem_converges_at_k_plus_one_in_u` therefore runs the whole `k =
1, 2, 3` study on **both** modes and requires the local orders to agree.

They do, to **1.8e-9, 3.2e-8 and 1.3e-7** at `k = 1, 2, 3` -- so the numbers above
are the iterative mode's own, and the table would be unchanged had it been the
exact mode's. The assertion is pinned at 0.01, the brief's "third digit of a
rate", rather than at the measured gap: the gap is set by where IDA's Newton
lands inside its own tolerance, which is not a portable quantity, while a real
disagreement between two solves of the same equations would be an order of 0.1 or
worse.

**`getFieldSweepStats().fallbacks` is zero at every refinement, on every case in
the file, and that is asserted rather than printed** (`checkNoFallbacks`). The
distinction is the whole finding: forcing an escalation by capping the sweep at
one iteration leaves every local order in the file *bit for bit unchanged* -- the
escalation returns the exact path's answer -- and drives the two-mode gap from
1.3e-7 to **exactly zero**, so the agreement check passes more strongly while the
study has stopped measuring the iterative mode at all. Only the fallback count
tells the difference, and only if something checks it.

The sweep runs 2.5 to 3.6 iterations per Jacobian solve here. Read that beside
`FieldJacobianTests.cpp`'s 13 to 38 sweeps on *random* right-hand sides, where
three of six exhaust the shipped cap of 20: the two are not in tension, and the
contrast is the explanation for why that cap is adequate in a real run. Newton's
right-hand sides are small, smooth corrections about a nearby state; a random
vector is the hard case. Neither number says anything about the cap on its own.

One outlier is worth recording so it is not mistaken for a defect later: the
multi-DOF case at `n = 8` takes 1591 field solves and 4877 sweeps where its
neighbours take ~200 and ~700. That is IDA working harder over that particular
step sequence, not the sweep failing -- the fallback count is still zero and the
ratio, 3.07 sweeps per solve, is inside the band every other row sits in.

**These tests are not vacuous, and that was checked rather than assumed.** Three
mutations, applied and reverted:

| mutation | effect |
|---|---|
| `SigmaFn` drops `s.geom(0)`, i.e. the geometry never reaches the physics | the flux check fails at every sample point; both single-DOF studies die with `IDASolve could not complete`; the multi-DOF orders collapse to -0.001, -0.000, -0.000 |
| `ManufacturedField::FieldResidual` becomes `psi - 1.05 Int u dx`, a 5% error in the field row -- not even a sign error | `k = 1` orders fall to 1.79, 0.85, 0.07; `k = 2` to 0.26, 0.00, -0.00; `k = 3` to 0.005, 0.000, 0.000; `psi` to -0.007 |
| `FieldSolveMaxSweeps` capped at 1, so every Jacobian solve escalates | every order unchanged, the two-mode gap improves to exactly zero, and only `checkNoFallbacks` fires -- 382 fallbacks in 382 solves at `k = 1, n = 4` |

The second is the one that matters for the equations: it is the failure mode the
whole file exists for, and the study loses the rate entirely rather than
degrading by an order. The third is the one that matters for the *method*, and it
is why the fallback count is asserted.

Note that the two solve modes still agreed -- to 1e-8 under the first two
mutations, and exactly under the third. That is the point of the cross-check
rather than a weakness in it: it is a statement about the linear solve, and
carries no information whatever about the equations.

### What the coupled study does not cover

* **`Superconvergent = true` on the multi-DOF model.** The flag is measured at
  `k = 1, 2, 3`, but only against `ManufacturedField`; nothing runs the star nodes
  against a geometry with five field unknowns behind it.
* **A settled rate for `psi`, or for `u*` at `k = 3`.** Both sweeps stop at
  `n = 32` and both are still moving there. Refuting a pre-asymptotic transient
  takes `n = 64`, as `the_flag_off_superconvergence_at_k2_is_genuine_not_pre_asymptotic`
  had to do for the uncoupled case.
* **A differential field DOF.** Both manufactured models are algebraic here.
  `CoupledResidualTests.cpp` runs the differential declaration end to end at one
  grid and compares its answer to the algebraic one, but nothing measures its
  order.
* **`nAux > 0` with a field.** Measured separately: `FieldJacobianTests.cpp`'s
  `GeometricAuxDiffusion` has both, but that is a Jacobian check, not an order
  study. `nScalars > 0` with a field is refused by `setFieldModel` outright.
* **A geometry that is not a smooth function of `x` within a cell.**
  `ManufacturedFieldVector`'s hat interpolant is only piecewise linear, so `g'`
  jumps at 0, 0.25, 0.5, 0.75, 1 and the manufactured source jumps with it. Every
  grid in the sweep is a multiple of 4, so those land on cell boundaries. A grid
  that put one *inside* a cell would lose the rate for a reason that has nothing
  to do with the coupling -- and the source-consistency check would still pass,
  since it samples away from the kinks by more than the stencil's reach.

### There is no coupled regression case, deliberately

`Tests/RegressionTests/` has no `coupled-field.conf` and is not going to get one
until a field model with physics in it exists. A regression case selects its
model by name from `[configuration] FieldModel`, which resolves against the
process-global registry, so the model would have to live in `PhysicsCases/` and
be linked into the shipped binary. The only two models that exist are
manufactured fixtures with no physics in them -- `ManufacturedField` and
`ManufacturedFieldVector`, both under `Tests/UnitTests` and both deliberately
unregistered -- and registering a fixture into the production binary to give the
regression suite something to point at is a worse trade than the gap it closes.

What covers the coupled path instead, and what each catches:

| test | what only it sees |
|---|---|
| `MMSFieldTests.cpp` (order study) | an error in the coupled *equations*: a sign or a factor in the residual, which converges at the right rate to the wrong function |
| `FieldJacobianTests.cpp` | an error in `A1` or `A2`, which costs Newton iterations and nothing else, since the coupled Jacobian is never assembled |
| `FieldAdjointTests.cpp` | an error in the *transpose* of either, which is a silently wrong gradient beside a perfectly good `G` |
| `SolverLifecycleTests.cpp::psi_round_trips_through_a_restart` | psi missing from, or misread out of, `<stem>.restart.nc`, and a resumed state that cannot be integrated on |
| `SolverLifecycleTests.cpp::a_coupled_solver_reused_matches_a_fresh_one_bit_for_bit` | a field model that caches across runs, or an `initialize()` that stops calling `resetForRun` |
| `SolverLifecycleTests.cpp::a_coupled_run_writes_the_field_group` | the netCDF group: its name, its `label`, and psi and geometry written to the wrong shape |
| `FieldModelSpecTests.cpp` (two name cases) + `CoupledResidualTests.cpp` (two collision cases) | a spec whose names netCDF cannot use, which is otherwise an `NcBadName`/`NcNameInUse` out of `ncGroup.cpp` at the first write |

**The restart case's oracle is the raw netCDF array, not `getSolution()`, and
that is load bearing.** `yJac` is filled by `DGSoln::copy` — the function this
work taught to carry the field block — so comparing `getSolution()` on both
sides of the round trip agrees perfectly when `psi_ = other.psi_` is deleted:
both are zero. Measured: with the oracle rooted in `getSolution()` that deletion
left the case *green*; rooted in `Y[nDOF - 1]` it fails three ways, the first
being `0.4421 != 0`. The vacuity guard has to exclude zero explicitly too — the
"psi actually moved away from `PSI0`" guard is satisfied by zero, since `PSI0`
is 0.5.

**The gap that leaves is real and is not covered by any of the above: nothing
exercises the coupled path through a `.conf` file.** The config plumbing --
`FieldModel` reaching `FieldModels::InstantiateFieldModel`, `FieldSolve` and the
three sweep keys reaching the solver through `applySolverConfig`, and the
restart branch of `runManta` shaping its `DGSoln` from `RestartData/nField` --
is covered by unit tests only. So is the netCDF group. The nearest thing to
end-to-end cover is the zero-coupling check below, which runs the binary over
every regression config and proves the coupling is *inert*, not that it works.

**The zero-coupling invariant is checked by hand, not asserted.** Every existing
config has no field model, so every one of them must produce output identical to
the same run built before the branch. Run each `Tests/RegressionTests/*.conf`
under both binaries from the same directory and `cmp` the results -- netCDF
files carry no timestamp of their own, so a byte comparison is legitimate, and
the regression suite's own 5e-3 is far too loose to see a change of this kind.
Measured at the point the serialisation landed: **all 14 `.nc` files byte
identical**, and all 14 `.restart.nc` files identical apart from the one
deliberately added `int nField = 0`.

## Superconvergence

`Superconvergent = true` switches the residual and Jacobian to the interpolatory
HDG_k scheme of Chen, Cockburn, Singler & Zhang (*J Sci Comput* 81:2188): the
physics is evaluated at the `k+2` nodes of the degree-`k+1` basis with the
postprocessed `u*` in place of `u_h`, and interpolated into `P_{k+1}` rather than
`P_k`. The reconstruction itself is `Postprocessing.{hpp,cpp}` and is built for
every run with `k >= 1`, flag or no flag, so `u_star` appears in the netCDF
output and in the `.dat` files unconditionally.

Four files carry this, in increasing order of scope.

**`PostprocessingTests.cpp`** pins the four per-cell operators. The anchor is
polynomial exactness: fed the `u_h` and `q_h` of a polynomial of degree `<= k+1`,
the reconstruction must return that polynomial. That single property fixes the
sign of the `A2` block (MaNTA carries `q = +d_x u`, the paper `q = -grad u`), the
scaling with `h`, and both `B11` and `B12` at once. Everything runs on a uniform
*and* a `High_Grid_Boundary` grid, because the operators are per-cell and a
mistake in the `h` scaling is invisible when every cell is the same size.

**`SolveJacTests.cpp`** finite-differences `residual()` with the flag on and
requires `solveHDGJac` to satisfy `J dy = g`. This is the only check that can
catch an error in the chain rule, since the Jacobian is never assembled and a
wrong one costs Newton iterations rather than accuracy. It uses a locally defined
`NonlinearDiffusion` whose flux *and* source depend on `u`; a constant-coefficient
case cannot see the `B11` term at all, which is the one genuinely new coupling
the scheme introduces. Measured `||J dy - g|| / ||g||` is 2e-10 to 5e-10 for
`k = 1, 2, 3`.

**`MMSConvergenceTests.cpp`** and **`MMSAuxScalarTests.cpp`** measure the observed
orders, flag off and flag on, for `u` and for `u*`. The shared sweep, the
least-squares fit and the exact solution live in `MMSHarness.hpp`; the first file
holds the problems with no couplings, the second the ones with `nAux > 0` or
`nScalars > 0`.

| case | k | flag off: u, u* | flag on: u, u* |
|---|---|---|---|
| linear, constant kappa | 1 | 1.96, 2.19 | 1.96, **3.05** |
| linear, constant kappa | 2 | 2.97, 4.08 | 2.97, 4.03 |
| nonlinear reaction `u^3 - u` | 1 | 1.96, 2.23 | 1.97, **3.07** |
| nonlinear reaction `u^3 - u` | 2 | 2.95, 4.10 | 2.97, 4.03 |
| nonlinear flux `(1+u^2) q` | 1 | 1.96, *2.81* | 1.92, **3.08** |
| nonlinear flux `(1+u^2) q` | 2 | 2.94, 4.20 | 2.92, 4.42 |
| aux `phi = u^2`, `(1+phi) q` | 1 | 1.85, *2.69* | 1.89, **3.18** |
| aux `phi = u^2`, `(1+phi) q` | 2 | 2.89, 4.12 | 2.89, 4.59 |
| algebraic scalar `mu = Int u dx` | 1 | 1.96, 2.16 | 1.96, **3.08** |
| algebraic scalar `mu = Int u dx` | 2 | 2.97, 4.07 | 2.97, 4.03 |
| differential scalar `mu' = Int u dx` | 1 | 1.96, 2.17 | 1.96, **3.08** |
| differential scalar `mu' = Int u dx` | 2 | 2.97, 4.07 | 2.97, 4.03 |

and the coupled quantities themselves:

| case | k | flag off | flag on |
|---|---|---|---|
| aux `phi` | 1 | 1.91 | 2.00 |
| aux `phi` | 2 | 2.78 | 2.98 |
| scalar `mu`, algebraic | 1 | 2.15 | **3.09** |
| scalar `mu`, differential | 1 | 2.26 | **2.96** |

Read the first table carefully, because it is not what a first reading of the
papers predicts. With the flag on, `u*` reaches `k+2` in every case and `u_h`
keeps its optimal `k+1`. With the flag off, `u*` superconverges at `k = 2` but
*not* at `k = 1`, and this is true whether or not the source is nonlinear. So the
flag restores the extra order at `k = 1` and preserves it at `k = 2`.

For the linear and reaction cases the loss is not driven by the nonlinearity, as
the papers' analysis of `I_h F(u_h)` would suggest. Interpolating a *known* smooth
source at the Chebyshev nodes leaves an error that is very nearly
`L2`-orthogonal to `P_k`, so it does not pollute the duality argument the way
evaluating `F` at the `O(h^(k+1))`-accurate `u_h` does.

**The two italicised flag-off entries are fits through a rate that is still
falling, and are the reason the flag earns its keep.** For the nonlinear flux at
`k = 1`, flag off, `u*` falls 4.55e-2, 6.58e-3, 5.60e-4, 6.18e-5, 2.74e-5 --
ratios 6.9, 11.7, 9.1, then 2.3. It superconverges over the coarse grids and then
stops. A sweep ending at `n = 32` reports 3.21 and looks superconvergent; the
2.81 above only appears because that case runs to `n = 64`. With the flag on the
same column falls by 8.5, 8.7, 8.4, 8.2 -- `2^3` every time. The aux case, which
is the same PDE with `(1+u^2)` routed through `phi`, behaves the same way. So for
a flux outside the papers' theory the postprocessing gain is real but transient
without the flag, and durable with it.

**That does not, however, explain away the `k = 1` / `k = 2` split in the linear
rows, and it was worth checking that it did not.** The obvious suspicion is that
the linear `k = 2` flag-off entry (4.08 over `n = 4, 8, 16`) is the same transient
caught before it breaks. `the_flag_off_superconvergence_at_k2_is_genuine_not_pre_asymptotic`
refines that sweep to `n = 64` and refutes it:

| n | flag-off u* | local order |
|---|---|---|
| 4 | 2.101e-04 | |
| 8 | 1.228e-05 | 4.10 |
| 16 | 7.387e-07 | 4.05 |
| 32 | 4.516e-08 | 4.03 |
| 64 | 2.739e-09 | 4.04 |

Four consecutive refinements at `k+2`, with none of the decay the nonlinear flux
shows by its third. So the two phenomena are distinct and both real: for a linear
flux at `k = 2` the interpolatory scheme keeps the extra order without the flag,
and for a nonlinear flux it loses it. The `k = 1` / `k = 2` anomaly stands
unexplained.

That last point is worth 2.7e-9, close enough to the 1e-9 relative tolerance to
ask whether it is measuring space at all, so the test carries a control:
loosening the tolerance tenfold moves `u*` from 2.739e-9 to 2.718e-9, about 1%,
where a tolerance-limited error would move by ten. `Tolerances` is a parameter of
`solveAndMeasureBoth` for exactly this purpose. **Prefer the local orders in the
per-`n` output to the single fitted slope** -- a fit averages a changing rate away,
which is how the nonlinear-flux breakdown hid at `n <= 32`.

The tests assert what is measured -- `u*` reaches `k+2` with the flag on, `u_h`
does not regress, and the coupled quantity keeps `k+1` -- and do not assert that
the flag improves on the flag-off rate, because for the linear problems there is
not always anything to improve.

**Scalars never see `u*`.** `ScalarG` and `ScalarGPrime` are evaluated on
`Y_h.evalOnNodes()`, the element nodes with `u_h`, in both the residual
(`SystemSolver.cpp:1194-1201`) and the Jacobian (`SystemSolver.cpp:815-818`),
regardless of the flag -- by design, since the scalars do not enter the
postprocessing. So a scalar cannot superconverge *through* the reconstruction,
and no test claims one does. What the second table shows is subtler and worth
keeping: `mu` still gains an order with the flag on, because it is a linear
functional of `u_h`, the flag changes what `u_h` is, and the functional error of
the flag-on solution is an order better than its `L2` error even though the two
solutions share an `L2` rate. The `k = 2` `mu` figures are omitted from that
table deliberately -- those errors reach 1e-8, within about an order of the 1e-9
relative integration tolerance, so the fits sit on the temporal noise floor and
the apparent `k+3` there is not a rate.

`phi` is capped at `k+1` and is not expected to do better: it is a `P_k` field
whatever it is constrained to equal, so interpolating a `k+2`-accurate `u*^2`
back into `P_k` gains nothing.

One trap worth stating, because it silently produces a vacuous test. The
`ManufacturedReaction` device of writing `Sources = f(x,t) - F(u)` so that the
compensating term vanishes at the exact solution **does not work against an
auxiliary variable**. `residual` evaluates `Sources` and `AuxG` on the same
states at the same abscissae and pushes both through the same
`projectOntoTestSpace` (`SystemSolver.cpp:1104`), so `Sources = S - G` gives
`S_cellwise = P(S) - res.Aux` exactly, and adding the term is precisely the row
operation `res.u += res.Aux`. The discrete solution set is unchanged at every
`h`, in both modes, and the study measures the uncoupled problem while looking
entirely healthy. `ManufacturedAux` therefore compensates against
`uExact(x,t)`, a known function of `x` and `t`, which cannot cancel against any
residual row.

### What is not covered

* **`k = 0`** is rejected (`std::invalid_argument`). The degree-0 `NodalBasis`
  returns from its constructor before building `Vandermonde` or
  `BarycentricWeights` (`Basis.hpp:369-377`), so it cannot be evaluated off-node.
  Paper I requires `k >= 1` for the superconvergence in any case.
* **Spatial adjoint parameters** are rejected with the flag on: they index the
  parameter vector by node, so the star node set would silently redefine how many
  parameters there are.
* **More than one aux variable or scalar at a time.** Every manufactured case
  here has `nAux <= 1` and `nScalars <= 1`, so a block-indexing error that only
  appears at the second one would not show. `SystemSolverMatrixTests.cpp` covers
  the layout with two scalars and `ScalarJacobianTests.cpp` with three.
* **`nAux > 0` together with `nScalars > 0`.** The two are measured separately.
* `dSourcedPhi_Mat` (`Matrices.cpp`) and the second `dAux_Mat` overload build
  their blocks by Gauss quadrature while the residual interpolates, so for
  `nAux > 0` they were *already* inconsistent with the residual before this work.
  The flag-on path uses interpolatory forms; the flag-off path is left exactly as
  it was so regression output stays bit-identical.

## Testing paired implementations

Several Jacobian builders exist in pairs -- one taking a pointer-to-member and
evaluating the physics per node, one reading precomputed batched values out of a
`GlobalStateMatrix`. The batched forms are what the solver calls; the per-node
forms are the older code, and `dAux_Mat`'s and `dSourcedPhi_Mat`'s now have no
callers at all. Keep them anyway: they are independent reference implementations
of the same block, and comparing the two is what the tests below do. Do not
reintroduce a call to either -- `dSourcedPhi_Mat` integrates by quadrature where
the residual interpolates, which is why `initializeMatricesForAdjointSolve` no
longer uses it.

Comparing the two is the single highest-signal test available for this code, and
it is how the `dAux_Mat` column-layout defect was found. Two things make it work:

* **`DerivativeSubMatrix`'s two overloads implement the same formula**
  (`MassMatrix * diag(f(nodes))`), so they must agree exactly for any state.
  `SystemSolverMatrixTests.cpp` asserts this for all five blocks on every cell.
* **`dAux_Mat`'s two overloads do not** -- one is interpolatory, the other
  integrates by quadrature. They coincide only when the derivative is constant
  across the cell, where both reduce to `c * M`. The mock therefore uses
  *constant* aux derivatives, which isolates the column layout from the
  quadrature scheme. Give them distinct primes (2, 3, 5, 7, 11, 13, 17) and a
  mis-slotted entry cannot come out right by accident.

## Known gaps

These are deliberate and tracked, not oversights:

* **Driving a C++ case from Python is pinned by an exact comparison against the
  TOML surface, and the gap is plugins.** `python/Tests/test_cpp_cases.py` runs
  two cases -- `LinearDiffusion` and `AdjointTestProblem` -- twice each, once
  from a config file it generates and once from the dict that file was generated
  *from*, and requires the netCDF output to be equal **bit for bit**. That is the
  whole strength of the file: both surfaces already share
  `loadSolverConfig`/`applySolverConfig`/`makeGrid`, so a difference can only be
  in the new half — a physics table that did not arrive, a float that became an
  integer, a grid handed over after construction — and none of those is small
  enough to hide under a tolerance. Compared through the netCDF rather than
  `getSolution` for a reason: both files are written from `y` at the same output
  times, while `getSolution` reads `yJac`, the state as of the last Jacobian
  evaluation, which can lag the final step and would force a tolerance back in.

  `AdjointTestProblem` is there to carry what `LinearDiffusion` cannot: **two**
  physics tables, a case deriving from `AutodiffTransportSystem` (whose
  constructor is the only in-tree code that actually *reads* the `Grid` it is
  handed -- `xL`/`xR`), boundary *kinds* set from a table rather than a
  coefficient, and `solveAdjoint = true`. `ADTestProblem` would have been the
  closer analogue and cannot be used: `Config/ADTestProblem.conf` does not run on
  `main` either, dying in `IDACalcIC` with `IDA_ERR_FAIL`. `Config/AuxVarADTest.conf`
  is likewise broken ("nAux > 0 but no coupling to fluxes provided"). Both
  predate this work; nothing in CI runs the `Config/` files, which is why neither
  is noticed.

  What is **not** covered: `load_physics_plugin` on a real plugin. Only the
  missing-file path is tested. A plugin needs `cmake --install` and a shared object
  compiled with the flags `pkg-config --cflags manta` reports, which is more than
  a pytest should build -- and it is the same gap the TOML surface's
  `PhysicsPlugins` key already has, so the `dlopen` flags and the
  do-not-link-`-lmanta` rule are covered by inspection on both surfaces. Nor does
  anything exercise a **restart** through a C++ case named from Python, or a case
  with `nScalars > 0`.

* **Degree adaptation is covered in three separable layers, and the gaps are in
  the physics it has been driven on rather than in the code.**
  `DegreeAdaptationTests.cpp` splits Giorgiani's rule (arithmetic, no solver),
  the accuracy indicator (a solution, no loop) and the driver (both), so a
  failure says which of the three moved; `python/Tests/test_degree_adaptation.py`
  covers the ownership change that only the Python surface has, where `run_ss()`
  replaces the solver `configure()` built.

  What no test reaches: **`nAux > 0` and `nScalars > 0`**. The indicator is
  formed per *variable* and the auxiliary variables are not looked at, which is
  a real question rather than a missing line — an under-resolved `phi` would not
  raise the degree. Nor is a **multi-variable** case exercised, so the "worst
  variable wins" reduction and the per-variable `Absolute_tolerance` floor are
  covered by inspection only. And nothing drives it with an **adjoint problem
  attached**, so `setAdjointProblem` being re-applied to each level's solver is
  untested — which matters, because forgetting it is silent: the run completes
  and the gradients are simply never computed.

  Two numbers worth keeping. On `AdjointPoster` at 6 cells the loop runs
  `k` = 2, 5, 8, 10 and takes the estimate 2.1e-3, 8.6e-6, 1.6e-8, 2.0e-10 --
  four solves under the default `MaxDegreeIncrement = 3`. Without the cap it is
  three solves, 2, 9, 10, reaching the same place with nothing to read in
  between.
  On `NonlinDiffTest` it climbs to the ceiling and stops, because that case's
  exact solution is `(1 - x/sqrt(t))^(1/n)` — a square-root branch point sitting
  on the upper boundary, inside the last cell — and the estimate falls like
  `1/k` exactly. That is the *right* answer on a problem p-refinement cannot
  fix, and it is the same regularity cap `MESH-REFINEMENT.md` records for
  Shestakov. Worth knowing before reading a climb to the ceiling as a defect.

* **A restart can change the polynomial degree but not the mesh.** Three cases
  in `SolverLifecycleTests.cpp` cover the degree change end to end -- refining
  reproduces the coarse state to 2.2e-16, coarsening lands on a cold run's state
  to the same, and equal degrees still take `DGSoln::copy` bit for bit. The
  exactness in the middle one is not luck: L2 projections onto nested spaces
  compose, so `P_{k-1}(P_k(f)) = P_{k-1}(f)`. That nesting is **per cell and
  needs the same cell boundaries**, so none of it carries over to a remesh, and
  a projection onto a different mesh would also want a merge-walk over the two
  boundary lists -- `DGApprox::operator()` finds its cell by linear scan, so the
  naive version is O(nCells^2 k).

  What is not covered: the `nDOF_file != nDOF` check is still written out
  separately in `MaNTA.cpp` and `PyRunner.cpp`, with the DOF formula duplicated
  in both, and no test drives either. It can only catch an
  `nVars`/`nAux`/`nScalars` mismatch, since the cell count and the file's degree
  both come from the file itself. Nor does anything drive `restartRunOrder`
  through a real config file -- the unit tests reach `setRestartValues`
  directly, so the config plumbing on both surfaces is covered by inspection
  only.

* **Adjoint *output* is still not verified.** `WriteAdjoints()` is commented
  out at `Solver.cpp:350` (commit `57d2652`, "adjoint writing doesn't work for
  spatial adjoints"), so no run emits the `ng` variable or the `G<i>_p` /
  `G<i>_boundary` groups. Both suites guarded their adjoint comparison on the
  *freshly generated* file containing `ng`, so the check silently skipped itself
  and could never fail -- even though `AdjointTestProblem.ref.nc` still carries
  `G0_p`, `G0_boundary`, `G1_p` and `G1_boundary`. Both suites now print a loud
  SKIPPING warning instead.

  The adjoint *gradients* are no longer unverified, though.
  `python/Tests/test_adjoint.py` reaches `G_p` through
  `PyRunner::getAdjointGradients`, which does not depend on `WriteAdjoints`, and
  checks it two independent ways for a problem with a closed-form solution:
  against central finite differences of the objective (re-running the solver at
  perturbed parameters) and against the analytic `dG/dp`. Both agree, so
  `initializeMatricesForAdjointSolve`, `solveAdjointState` and
  `computeAdjointGradients` are now covered end to end. What remains untested is
  serialising them to netCDF.

  `python/Tests/test_adjoint_aux.py` does the same at `nAux = 2`, `nVars = 1`.
  That intersection -- adjoints *and* auxiliary variables -- was uncovered, and
  three defects lived in it, all fixed:

  * `initializeMatricesForAdjointSolve` never wrote the `dSigma/dPhi` block, so
    the stored matrix was not the transpose of the forward Jacobian whenever the
    flux depended on `phi`. The fixture routes the *entire* flux/derivative
    coupling through that block (`sigma_hat = kappa*phi_q` with `phi_q - q = 0`,
    so `dSigmaFn_dq == 0`), which makes the adjoint operator singular without
    it: the gradient comes out `-4e14` instead of `-9.9e-3`, while `G` itself is
    unaffected. That asymmetry is the point -- a bad forward Jacobian costs
    Newton iterations, a bad adjoint matrix costs correctness silently.
  * the same function used `dSourcedPhi_Mat` (quadrature) where the residual
    interpolates; it now uses `dPhi_Mat`, as the forward Jacobian already did.
  * `dGdaux_Vec` sized two things by `nVars` that are indexed to `nAux` -- its
    output-length assert, and the scratch vectors passed to `dgFn_dphi`. Both
    are latent at `nAux == nVars`, which every other aux fixture has.

  `nAux != nVars` is deliberate in that fixture: it is what distinguishes the
  two lengths. The extra auxiliary variable (`phi_u - u = 0`) is otherwise
  unused.

* **The coupled adjoint is C++-only, and the Python surface cannot reach it.**
  `Tests/UnitTests/FieldAdjointTests.cpp` covers the field coupling's adjoint end
  to end -- against a closed-form `dG/dp`, against finite differences, and
  against the transpose of a finite-differenced coupled Jacobian -- but there is
  no Python equivalent, and that is a gap in the *bindings* rather than in the
  tests. `FieldModel` has no pybind11 class, so a Python case cannot define one;
  the `FieldModel` configuration key names a *registered* model, and no field
  model is registered anywhere in the tree (both manufactured ones live under
  `Tests/UnitTests` and are deliberately unregistered). So there is nothing a
  `python/Tests/test_adjoint.py` fixture could attach. Adding the coupled Python
  check means first registering a production field model or binding `FieldModel`
  to Python -- neither of which the adjoint work owns.

  Two limits of the coupled adjoint itself are structural rather than untested,
  and are recorded in `TODO` and beside `G_field` in `SystemSolver.hpp`: an
  objective whose integrand reads `State::geom` directly loses its `dG/dpsi`
  term, because `AdjointProblem` reports four state derivatives and geometry is
  not among them; and a `FieldModel` cannot depend on an adjoint parameter at
  all, so `d(field residual)/dp` is zero by construction.

  One defect that fixture found was **not** in the coupling. The adjoint's local
  matrix stored `+Sq` where the forward Jacobian builds `-Sq` -- the `u` row's
  `q` column, i.e. `dSources_dq` -- so `initializeMatricesForAdjointSolve` had
  not been the transpose of `assembleCellMatrix` for any case whose source reads
  `q`. Every adjoint fixture in the tree has `dSources_dq` identically zero, so
  `Sq` was the zero matrix and its sign never mattered; the coupled fixture
  carries `dSources_dq = 0.2` and put the gradient 0.48% out. The `J^T z = g`
  check is what localised it to the operator rather than to `F_p` or to a state
  that was not quite steady.

* **`python/Tests/test_reference_solutions.py::test_jax_aux_test` passed and the
  xfail is gone.** It had been `strict=True` xfail since `fdd5ee1`. The
  conclusion the note above reached -- that the C++ `nAux > 0` path was sound
  and the fault lay in `manta.jax` -- was right, and there were four faults, all
  reachable only with `nAux > 0`, which no other JAX fixture has:

  * `AuxGPrime(i, out, state, x, t)` and `dAux_dp(i, pIndex, state, x)` carry an
    extra argument *ahead of* the state, and both were decorated with
    `MaNTA_Decorator`, whose wrapper is `(self, index, states, positions,
    *args)`. So it converted the extra argument as though it were the state and
    passed the state to `jnp.array()`. They take `ShiftedState_Decorator` now.
  * `dgFn_dphi` was not decorated at all, so it was handed a raw `manta.State`
    inside a `jax.jit`, and it indexed its result `["Aux"]` -- left from when a
    State crossed as a dict, where `jax.grad` now returns the State module.
  * `JAXAdjointProblem.__init__` bound `sigma` and `source` but not `aux`, so
    `dAux()` raised `AttributeError` on the third branch of
    `ComputePhysicsDerivatives`.

  Two things about the diagnosis are worth keeping. The symptom the mark
  recorded -- "IDA's corrector will not converge at t = 0" -- was **not** the
  symptom by the time it was fixed; it had become a `TypeError` out of
  `AuxGPrime`. The first fault hit changes as the surrounding code moves, so an
  xfail *reason* ages badly in a way an xfail *mark* does not, and re-running
  before theorising was what showed it. And the reference the run is compared
  against was generated in February, before any of this broke: the solution
  reproduces it to 1e-2 relative L2 at every output time, so the fixture really
  was innocent all along.

  `python/Tests/test_jax_aux.py` now catches three of the four directly, against
  hand-differentiated closed forms rather than against another autodiff run. The
  fourth, `dgFn_dphi`, is gone from `manta.jax` since: `dg/dphi` arrives with the
  rest of the batched `dg`, and `PyAdjointProblem::dgFn_dphi` raises rather than
  dispatching to Python at all.

* **`AuxG_v` was bound to the pointwise `AuxG`** (`Python.cpp`), where
  `SigmaFn_v` and `Sources_v` name the batched `(GlobalState, positions)`
  overload. That made it an exact duplicate of `AuxG` and left the aux path the
  only one a test could not drive through the C++ serial loop -- which is how
  the pointwise hooks are exercised from Python, a `State` having no constructor
  there. Fixed, and pinned by `test_aux.py::test_auxg_v_is_the_batched_overload`.

* **A second integration on the same `SystemSolver` now works**, and
  `SolverLifecycleTests.cpp::a_second_integration_on_one_solver_matches_a_fresh_one`
  pins it -- three consecutive `runSolver()` calls on one object, compared bit for
  bit against a fresh solver. It used to fail with `IDA_ERR_FAIL` (-3) on the
  first step of the second run. Two defects combined, and it needed both:

  * `id` was all zeros, so IDA was told the whole system was algebraic and
    `IDA_YA_YDP_INIT` solved the wrong initialisation problem; and `IDACalcIC`'s
    return value was discarded, so a failure there carried on into the time loop
    from IDA's partial state. Since the algebraic components are in IDA's error
    test and their error estimate is then independent of `h`, no step was small
    enough to pass -- ten error-test failures, then `IDA_ERR_FAIL`. Both fixed in
    `536d856`, which is *after* the note recording the failure was written
    (`c9f175d`); nobody re-tested it.
  * What made the second run differ from the first: `initialiseMatrices` filled
    `RF_cellwise` and `L_global` at a hardcoded `t = 0.0` and `initialize()`
    skips it when already initialised, so a re-initialise solved its initial
    `dydt` out of the *previous run's final-time* boundary data. Those arrays are
    now sized there and filled by `updateBoundaryConditions(t0)`.

  The second defect is worth knowing about beyond this bug, for two reasons.
  With the `id` fix alone the second run completed and looked right to any
  reasonable tolerance -- it was off by 1.7e-10, which is why the pinning test
  compares bit for bit rather than approximately. And the hardcoded `0.0` was
  independently wrong for any run with `t0 != 0`;
  `the_initial_condition_uses_boundary_data_at_t0` covers that separately,
  because every other fixture in the tree starts at zero.

* **Warm starts: what a restart hands the solver, and whether `IDACalcIC` has to
  run.** Two cases in `SolverLifecycleTests.cpp`, and unlike the three
  degree-transfer cases beside them they go through an actual `.restart.nc`
  rather than through `setRestartValues` on an in-memory vector -- the netCDF
  round trip is part of what decides whether the state is still consistent.

  * `the_warm_start_keeps_the_trace_the_file_carries`. `setInitialConditions`
    finished every restart with `EvaluateLambda()`, which sets `lambda` to
    `{{u}}` -- the average of the two cell traces -- and that is not the equation
    `lambda` solves. On a restart it discarded a converged trace and replaced it
    with something that solves nothing. The test measures both: keeping the trace
    gives a weighted residual of 2.6e-3, re-averaging it gives 556, a factor of
    2e5. That is the whole of the gap, and it is what made a restart need about
    ten times as many residual evaluations inside `IDACalcIC` as a cold start --
    the effect `TestSolutions.py`'s comment beside `check_restart_round_trip` had
    already noticed and attributed correctly. On `AuxVarTest`'s round trip
    keeping the trace takes the resumed run from 1139 residual evaluations to
    1033.
  * `a_warm_start_from_a_restart_file_does_not_run_calcic`. `IDACalcIC` has no
    cheap path -- 2 residual evaluations, 2 Jacobian builds and 2 Jacobian solves
    even on a state it just converged to, because its test is on the Newton step
    -- so a restart skips it by default. The test sets no key at all, checks
    `IDAGetNumResEvals` is zero after `initialize()`, **and integrates**, because
    nothing else establishes that the state can be stepped from. It separately
    checks the warm start's residual really was small, so the integration is not
    passing for a reason nobody chose; its control is a cold start of the same
    problem, three orders further out, which is still corrected; and it finishes
    by setting `ForceConsistentIC` on the same restart and requiring `IDACalcIC`
    back, which is the only thing that would notice the key being ignored.

  * `only_a_copied_restart_is_treated_as_already_consistent`. The default is
    conditional on the transfer having been a *copy*. A restart onto a different
    degree is projected -- `setInitialConditions` moves `u`, `q`, aux and the
    scalars and then rebuilds `sigma` and the trace -- so it is a guess like any
    other, and skipping there is a broken run rather than a saving: `AuxVarTest`
    resuming at a lower degree fails with `IDA_ERR_FAIL` when `IDACalcIC` is
    skipped and completes when it runs. Both halves are asserted, because without
    the second the projection path would quietly start from an inconsistent state.
    Note that the three degree-transfer cases beside it only `initialize()` and
    compare state; none of them integrates, which is why this was not already
    covered.

  What is *not* covered, and is why the key is a boolean rather than a threshold:
  **a residual norm is not the quantity that decides the question.** It replaced
  `ConsistentICTolerance`, which compared the initial weighted residual against a
  number the caller supplied. `IDACalcIC`'s own test is `||J^-1 F||_wrms` -- a
  correction to `y` -- and the two differ by the
  per-row amplification `s_i = ||J^-1 e_i||_wrms`, which is nowhere near
  proportional to the error weights. Measured as `s_i / ewt_i` on three cases: the
  `u` rows are over-weighted by up to ~4000x, the Dirichlet `lambda` rows carry
  the largest weight in the vector on rows whose sensitivity is *exactly zero*
  (`residual` never writes them), and the `aux` rows are under-weighted by up to
  ~10x relative to `sigma`. What that costs is calibration: over six `AuxVarTest`
  warm-start states -- three tolerances, corrected and not -- `||J^-1 F|| / ||F||`
  runs from **15 to 187**, for one problem at one discretisation.

  It used to be worse than uncalibrated. Before `AuxVarTest`'s missing
  `dSigma_dPhi` block was declared, `||F||` made that warm start's uncorrected
  state look 2.4x *better* (1.6e-4 against 3.8e-4) where `||J^-1 F||` made it 6.3x
  *worse* (2.0e-2 against 3.1e-3) -- and the run agreed with the second, the
  failing Newton's correction plateauing at 1.98e-2 as `h` fell. `||J^-1 F||`
  predicted that using the *defective* `J`, which is the point of it: it measures
  the Newton the solver will actually run. With the block declared every round
  trip completes with the skip armed and the two norms order those states alike,
  so there is now **no fixture in the tree that requires `IDACalcIC`** -- which
  removes the counter-example without establishing that any threshold is safe.
  Hence a boolean: a caller restarting from a state they know is converged has
  information the norm does not, and the key is how they say so.

  A correct test costs one residual, one Jacobian build and one Jacobian solve
  against `IDACalcIC`'s floor of two of each: a factor of two, not the near-free
  test this is. And the floor is all the skip saves -- measured on an already
  consistent warm start, 2 residual evaluations and 2 Jacobian builds, which is 2
  of 89 and 2 of 21 on `AuxVarTest` at rtol 1e-6.

  The opposite failure -- a `TestDiffusion` warm start at rtol 1e-6 that could not
  complete `IDACalcIC` at all -- **was a MaNTA bug and is fixed**. `initialize()`
  passed `IDACalcIC` the interval `dt` where `tout1` wants an absolute time; every
  fixture starts at `t0 = 0` where the two agree, and that one restarts at
  `t0 = 0.05` with a cadence of `0.05`, so `tout1` landed exactly on `t0` and IDA
  refused the input. It was never about the tolerance -- it reproduced at every
  tolerance, and with `tout1 = t0 + dt` that warm start converges in 3 residual
  evaluations, as does a degree-projection restart at a weighted residual of
  8.7e3. Two cases had been written around the belief that it was a hard state;
  both are corrected, and the fail-open case below now says so.

* **`SteadyState.cpp` is barely covered, and what covers it now is its output
  rather than its algorithm.** Until recently nothing called `solveSteadyState`
  from a test at all -- the only mentions of it under `Tests/` were config
  parsing in `ConfigSourceTests.cpp` and the tolerance setter in
  `SolverPlumbingTests.cpp`. Two cases in `SolverLifecycleTests.cpp` now drive it
  end to end, both provoked by defects rather than written for coverage:

  * `a_steady_solve_writes_its_answer_to_the_output_file`. Every output call
    lived inside the time loop, which `PseudoTransient` and `Newton` skip, so a
    steady run's `.nc` held exactly one timeslice -- the `t0` one
    `initialiseNetCDF` writes during `initialize()`, i.e. the *initial
    condition* -- and its `.dat` one block of the same. The converged state
    reached `yJac` and the restart file's `Y`, so `getSolution` was always right;
    only the files were wrong, and every steady run in this tree is driven from
    Python, which reads `yJac`. `writeDiagnostics` is called from
    `WriteTimeslice` and nowhere else, so a physics case's per-slice diagnostics
    were never called at all while `initialiseDiagnostics` and
    `finaliseDiagnostics` both ran -- the scaffolding at both ends with nothing
    hung on it. The converged state is now stamped
    `SystemSolver::STEADY_STATE_TIME` (1.0, a label rather than a time -- see
    `docs/running.rst`), and the test checks it against the closed form `u = 1-x`
    as well as requiring the two slices to differ.
  * `a_converged_steady_state_leaves_no_stale_derivative`. `solveSteadyState`
    damps through a scratch vector and never wrote back to `dYdt`, so on return
    it still held the `t0` derivative `IDACalcIC` left -- 103.4 in norm on
    `AdjointPoster` at a converged steady state, where the defining property of
    the answer is that it vanishes. `WriteRestartFile` and `writeDiagnostics`
    both read it, so a restart from a steady run resumed with a `y` and a `y'`
    that did not belong together. The check is exactly zero, not merely small: it
    is set rather than converged to.

  More cases have landed since, each also provoked by a defect:

  * `the_SER_rate_and_floor_change_the_cost_and_not_the_answer` measures the schedule through physics evaluations -- 552 at the defaults, 3540 with the floor at 1, 1704 with the floor at 1 and the rate at 2 -- and requires the converged state to be identical in all three. An option that changed the answer would be a bug; one that changed nothing would be inert.
  * `the_steady_diagnostics_count_the_whole_solve_not_the_last_step`. **KINSOL zeroes its own counters at the top of every `KINSol` call**, so the continuation loop has to sum them as it goes; reading them once at the end -- the obvious thing, and what this did first -- reported 1 Newton iteration against 5 continuation steps and 35 Jacobian solves. Self-evidently impossible, and it still looks like a number, which is why the test asserts invariants (`newtonIters >= steps`, `residualEvals == kinFuncEvals + steps + 1`) rather than values. The second of those also pins the counter snapshot being taken before the first `steadyNorm()`, which it was not to begin with.
  * `a_failed_steady_solve_still_writes_the_last_state_it_reached`, using a tolerance nothing can reach so the solve stalls at ~1e-16 and exits by the "ran out of continuation steps" path. It also checks the per-step trace survives the throw and is complete, since a run that failed is the one whose trace is worth having.
  * `the_per_step_records_sum_to_the_totals`. The per-`KINSol` records and the totals are gathered by two different routes -- the totals difference MaNTA's monotonic counters across the whole solve, each record differences them across one step -- so their agreement is a check rather than a restatement. It is what would catch a step whose record was never closed, and two of the three exits from the loop body are a `return` and a `throw`, so that is not hypothetical. The offsets are asserted exactly: `sum(jacBuilds)` and `sum(jacSolves)` *equal* the totals because nothing builds or solves outside the loop, while `sum(residualEvals) + 1` is the total, the one being the merit evaluation made before any step exists to charge it to. It also pins the records being cleared per solve rather than accumulated, which `PyRunner` depends on.
  * `newton_jacobian_reuse_trades_builds_for_solves` and `newton_max_iterations_caps_every_inner_solve`, which read the two KINSOL settings back through the diagnostics rather than through KINSOL -- there is no `KINGet` for `msbset`, and behaviour is the thing worth pinning anyway. Both need a **nonlinear** fixture, so `SolverLifecycleTests.cpp` carries a small `NonlinearDiffusion` (`sigma = (1 + u^2) q`): on `TestDiffusion` every inner solve converges in one Newton iteration, builds equal solves at every setting, and a reuse test would pass while measuring nothing. That degeneracy is checked for explicitly rather than assumed, so a change to the fixture that quietly made it linear fails the test instead of hollowing it out. Measured: reuse 1 gives 16 builds for 16 solves, reuse 10 gives 7 for 29, and the two agree on the answer to 1e-8. The iteration cap is read per step rather than in total, because a total could be held down by the solve simply needing fewer iterations, where a per-step maximum of exactly the cap can only come from the cap binding.
  * `the_newton_settings_refuse_values_that_cannot_work`. Zero iterations cannot make progress, and zero *reuse* is KINSOL's "use the default" sentinel -- so passing it through would silently mean 10 rather than what was asked. Zero is meanwhile legitimate for the step tolerance, where KINSOL implements exactly that meaning, and the test pins both readings of zero so they cannot be regularised into one.
  * `the_per_step_diagnostics_print_without_the_summary`. `SteadyStateStepDiagnostics` and `SteadyStateDiagnostics` are independent; this pins the direction that is easy to get wrong, since a trace implemented as extra detail inside the summary block would make the more specialised request unreachable without the less specialised one. Rows are counted from the outcome column rather than by counting lines, so an unrelated line elsewhere in the run cannot make it pass.
  * `only_a_time_marching_run_pays_for_calcic` and
    `skipping_calcic_leaves_the_steady_answer_alone`. `initialize()` ran
    `IDACalcIC` unconditionally, including for a solve that never takes an IDA
    step and so discards its answer with the first accepted continuation step.
    Worse than wasted: `IDACalcIC` is a damped Newton solve in its own right and
    fails on initial conditions the steady solver handles easily --
    `python-examples/jardin-critical-gradient` records `IDA_CONV_FAIL` (-4) from
    starting at the *exact* steady state, which is the one guess a steady solve
    would have taken instantly. The first test reads `IDAGetNumResEvals` straight
    after `initialize()`, which is zero unless something asked IDA to solve;
    MaNTA's own `nResidualEvals` would not do, since the debug `.dat` blocks and
    the steady solve increment it too. It covers three configurations, because
    the condition is `solvesForSteadyState()` -- the *conjunction* of armed
    termination and a non-`TimeMarch` mode -- and the trap is reading the default
    `PseudoTransient` as a steady solve when nothing armed termination. Mutating
    the gate to a constant fails it in both directions. The second test pins the
    thing that actually matters: what the solve converges to must not depend on a
    correction that was going to be thrown away, checked at both steady modes
    against `TestDiffusion`'s closed form. Measured on the benchmarks, this cut
    physics evaluations per point from 15 to 11 (`PseudoTransient`) and 11 to 7
    (`Newton`) on `park-convergence`, 142/167 to 138/163 on
    `jardin-critical-gradient` and 657/683 to 622/648 on `shestakov-nonlinear`,
    with every converged answer identical bit for bit and `TimeMarch` untouched.

  What is still uncovered is the rest of the algorithm -- step rejection, the
  `KINSetMaxNewtonStep` clamp, and the hard-`KINSol`-failure
  path (the ordinary exhaustion path above shares its `catch (...)` in
  `Solver.cpp`, but not the code that reaches it). In particular the flat
  unweighted `steadyNorm` (`SteadyState.cpp`) is untested, and both the
  convergence test and the SER ratio read it, so anything that changes how it is
  normalised changes the stopping test and the step schedule together.

  Note also what the fixture cannot show: `TestDiffusion` is *linear*, so each
  inner solve converges in one Newton iteration and Jacobian builds equal
  Jacobian solves. The separation those two counters exist to report only
  appears on a nonlinear problem -- `AdjointPoster` at `k = 3` on 6 cells pays 7
  builds against 35 solves. An assertion of `builds < solves` was written here
  first and failed for exactly that reason.

* **The C++ mirror plasma is gone.** `PhysicsCases/MirrorPlasma.{cpp,hpp}`,
  `PhysicsCases/MirrorPlasma/` and `PhysicsCases/CurvedMirrorPlasma/` were
  removed in favour of `python-physics/mirror-plasma`, which is the
  implementation that is developed now. `MirrorPlasmaTest.cpp` went with them:
  two cases, `plasma_init_tests` and `neutral_model_tests`, both of which
  constructed a `PlasmaConstants` and checked collision times and neutral rates
  against hand values. Nothing replaces them here — the equivalent checks are
  `python-physics/mirror-plasma/test_mirror.py`, which is not run by
  the `python` test (`pytest.ini` is `testpaths = python/Tests`) and needs
  `desc` and `optimistix`. That is a real reduction in what CI covers, recorded
  here rather than left to be discovered.

  `CurvedMirrorPlasma/` had never compiled in any case (commit `c17fa42`,
  "start to add in curved stuff (doesn't compile)"): 49 errors, including
  references to a `CurvedMagneticField` class and a `PlasmaTypes` enum that were
  never written. It was excluded from the build's physics sources for that
  reason, and it
  depended on `MirrorPlasma` and `PlasmaConstants`, so it could not have
  outlived them.

* **`PhysicsCases/` is reported but not gated.** It is exercised as test
  fixtures rather than as a coverage target in its own right.

## Threading

The physics is never threaded — the batched wrappers that fall back on pointwise
hooks are serial loops whatever `MANTA_OPENMP` says, because a case that supplies
only pointwise hooks never agreed to be called concurrently, and because a Python
case's GIL makes it more than 13x *slower* rather than faster. Only the solver's
own cell loops are parallel. `docs/physics_interface.rst` states the rule and
`CLAUDE.md` has the measurements.

`UtilityTests.cpp` carries four `parallel_for` cases. Three of them are near-
tautologies in an ordinary build — without `MANTA_OPENMP` the helper is a plain
loop — and they are there because the thing they pin cost a **process abort**: an
exception thrown by a physics hook inside an OpenMP loop reached
`__cxa_call_terminate` instead of `static_residual`'s handler, killing the whole
suite. `an_exception_from_the_body_reaches_the_caller` throws from the last index
deliberately, because the original defect passed whenever the throw landed on the
master thread.

**They only bite in a build that sets the option.** Until 2026-08-25 nothing did
— no CI leg, no preset — which is how the abort survived. `ci.yml` now has a
`Build + tests (g++-15, OpenMP)` leg, and it is in the branch-protection required
list, so a red one blocks a merge.

To run them by hand:

```sh
cmake -B build-omp -DMANTA_OPENMP=ON && cmake --build build-omp -j 6
OMP_NUM_THREADS=6 MKL_NUM_THREADS=1 ctest --test-dir build-omp --output-on-failure
```

`MKL_NUM_THREADS=1` is not optional on a box whose BLAS threads itself. With it
unset, `OMP_NUM_THREADS=6` also threads the BLAS, and the changed reduction order
was enough to fail `afn_tests/the_jacobian_agrees_with_the_residual_for_a_nonunit_coefficient`
with `IDACalcIC could not complete` — in a configuration where **none of MaNTA's
own loops were parallel at all**, since that test's 3 cells and 12 physics points
are both below the grain floors. Confirmed by separating the variables:
`OMP_NUM_THREADS=6 MKL_NUM_THREADS=1` passes, `OMP_NUM_THREADS=1
MKL_NUM_THREADS=4` fails. Worth remembering before attributing any threaded-build
failure to a race.
