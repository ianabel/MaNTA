# MaNTA test suites

Three suites, all driven from the top-level Makefile and all runnable from any
working directory.

| Command | Suite | Location |
|---|---|---|
| `make test` | Boost.Test C++ unit tests | `Tests/UnitTests/` |
| `make regression_tests` | Solver run against checked-in `.ref.nc` references | `Tests/RegressionTests/` |
| `make python_tests` | pytest suite for the pybind11 module | `python/Tests/` |

The Python suite writes its solver output into the *current directory*, not into
pytest's `tmp_path`: `OutputFilename` is passed to `setInputFile`, and
`Solver.cpp` keeps only `inputFilePath.stem()`. Tests that need output files
therefore use unique cwd-relative names and clean up after themselves; see
`test_output_filename_keeps_only_the_basename` in `test_runner.py`, which pins
the behaviour.

`make coverage` rebuilds with `--coverage -O0`, runs all three, and writes
`coverage/index.html` (numerical core + Python bindings) and
`coverage/physics.html` (`PhysicsCases/`, informational). See the README.

## Unit tests

Boost.Test in header-only mode (`boost/test/included/unit_test.hpp`), so there
is no `-lboost_unit_test_framework` link step -- `libboost-dev` is enough.
`Tests/UnitTests/main.cpp` defines the module; every other `.cpp` listed in
`TEST_SOURCES` contributes suites.

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
  which resolves against the `TEST_DATA_DIR` baked in by the Makefile. Do not
  hardcode `./Tests/UnitTests/...`; that only works from the repo root.

Run a single suite or case directly:

```sh
Tests/UnitTests/UnitTests --run_test=dg_approx_tests/grid_test --log_level=all
```

## Regression tests

`TestSolutions.py` runs the solver over each `.conf` and compares in four modes:
against a closed-form analytic solution, against a steady state, against a
checked-in `<name>.ref.nc`, and -- for three cases -- against itself via a
restart round trip. Tolerance is 5e-3, overridable with `--tolerance`.

The restart round trip runs to `t_final` in one go, then again in two halves via
a `.restart.nc` file, and requires the final states to match. It is the only
test of `WriteRestartFile` -> `StoreGridInfo` -> the restart branch of
`runManta` -> `setRestartValues`, and it is what would catch a repeat of the
clustered-grid contiguity defect (a grid rebuilt from a restart file must
compare *equal* to the one that wrote it).

**Restarting is fragile at tight tolerances**, so each case runs at the tightest
one at which it completes -- see the measured table beside the calls in
`TestSolutions.py`. Briefly: `LinearDiffusion` survives 1e-6, `MatTest` 1e-4,
`AuxVarTest` only 1e-3, while the *uninterrupted* run succeeds at every one of
those. After a restart `IDACalcIC` needs roughly ten times as many residual
evaluations as from a cold start, which points at `setInitialConditions`
recomputing sigma and lambda from the restored u and q while discarding the
restored dY/dt.

`.ref.nc` files are committed -- `.gitignore` has `*.nc` with a `!*.ref.nc`
negation.

**`AuxVarTest.ref.nc` was regenerated** after the `dAux_Mat` column-layout fix
(see the paired-implementations section below). The old reference was produced
by `f35e3ee`, the commit that introduced the defect, so it recorded the
behaviour of a run whose Jacobian was missing `dG/du` entirely. The case runs at
`Relative_tolerance = Absolute_tolerance = 1e-2`, loose enough that a different
Newton path moves the answer by about half a percent -- which is what tripped
the 5e-3 comparison.

The new output is *closer to the truth*, not merely different. Against a
tighter-tolerance run of the same discretisation (rtol = atol = 1e-5, which is
as tight as this case will integrate):

| | Var0 | Var1 |
|---|---|---|
| old reference | 1.56e-2 | 2.72e-2 |
| new output | 1.01e-2 | 1.37e-2 |

The pre-fix reference is kept out of the tree; recover it from
`git show HEAD:Tests/RegressionTests/AuxVarTest.ref.nc` if the comparison ever
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

## Open question: the scalar (Woodbury) path in solveJacEq

`SolveJacTests.cpp` builds the Jacobian by finite-differencing `residual` under
IDA's convention (`J = dF/dY + cj dF/dY'`) and requires the vector the linear
solve returns to satisfy `J dy = g`. With `nScalars = 0` this passes at **3e-10**,
so `solveHDGJac`'s static condensation is verified.

With `nScalars = 3` (`ScalarTestLD3`) the same check comes out **O(1)** -- about
0.3 in the DG field rows and 5-8 in the scalar rows. Because the error is not
confined to the scalar rows, there are two candidate explanations and they have
not yet been separated:

1. the Woodbury/bordered elimination in `solveJacEq` is wrong; or
2. `ScalarTestLD3::ScalarGPrimeExtended` disagrees with its own
   `ScalarGExtended`, which would corrupt the entire bordered solve rather than
   just the scalar rows -- which is what the numbers look like.

**The `PIDTest` regression case cannot distinguish these and does not contradict
either**: a wrong Jacobian only degrades Newton convergence, it still converges
to the correct answer, so a reference-output comparison stays green.

To settle it, add a minimal transport system with `nScalars = 1` and
`G = mu - const`, whose Jacobian is exactly known, and run the same check. If
that passes, the elimination is fine and the fault is in `ScalarTestLD3`.

Until then the test asserts only what is actually established (the solve
completes and returns finite values) and prints the residuals; the strict bound
is deliberately not asserted, so the suite is green but not claiming this path
is correct.

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
  case imposes.** `sin(pi x)` vanishes at both ends for every `t`. The `UseMMS`
  option on `LinearDiffusion` does *not* have this property -- its manufactured
  solution is the initial Gaussian, which is about 0.29 at the domain edge with
  the default parameters, while the boundary condition is 0 -- so it cannot be
  used for an order study as configured. (`LinearDiffSourceTest` reads `useMMS`
  and never applies `MMS_Source` at all.) Neither is exercised by any regression
  case; both are recorded in a test case rather than changed.
* **The time-integration tolerance must be well below the spatial error, but not
  so tight that IDA cannot start.** At 1e-12 it fails at `t = 0` for `k >= 2`
  with "the error test failed repeatedly or with |h| = hmin"; 1e-9 leaves three
  orders of margin over the smallest spatial error in the sweep.

## Testing paired implementations

Several Jacobian builders exist in pairs -- one taking a pointer-to-member and
evaluating the physics per node, one reading precomputed batched values out of a
`GlobalStateMatrix`. The batched forms are what the solver calls; the per-node
forms are the older code and, in `dAux_Mat`'s case, now have no callers at all.

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

* **`python/Tests/test_reference_solutions.py::test_jax_aux_test` is xfail
  (strict).** The Python `nAux > 0` path returns demonstrably correct
  derivatives but IDA's corrector will not converge at t=0. Ruled out: float32
  precision, tolerance/polynomial degree/grid size/timestep, and
  source-constraint degeneracy. The fixture is unchanged since `1a369d7`
  (2026-02-05), when it generated the committed reference, so this is a
  regression in the aux path rather than a bad fixture. `strict=True` means the
  suite will fail if it starts passing -- that is the signal to remove the mark.

  **Narrowed since.** Two candidate causes have now been eliminated:

  * the `dAux_Mat` column-layout defect (which dropped `dG/du` from the
    Jacobian entirely) is fixed, and the xfail is unchanged;
  * `python/Tests/test_aux.py` runs the *same* reaction-diffusion problem with
    `nAux = 1` through the same C++ trampoline in plain numpy, and it converges
    and satisfies `a = u^2` against an independent Newton solve.

  So the C++ `nAux > 0` path is sound and the fault is specific to the JAX
  fixture or to `JAXTransportSystem`'s aux hooks. That is where to look next.

* **`PhysicsCases/CurvedMirrorPlasma/` is excluded from the build.** It is
  unfinished (commit `c17fa42`, "start to add in curved stuff (doesn't
  compile)") and has never compiled: 49 errors, including references to a
  `CurvedMagneticField` class and a `PlasmaTypes` enum that were never written.
  Adding it to `PHYSICS_SOURCES` breaks `make` for everyone.

* **`PhysicsCases/` is reported but not gated.** It is exercised as test
  fixtures; `MirrorPlasma` and the plasma diagnostics are not a coverage target.
