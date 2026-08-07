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

A run writes netCDF (`<stem>.nc`, `<stem>.restart.nc`) by default and no text
output at all; `.dat` files need `WriteDatFile` (and `WriteDebugDatFiles` for
the `.dydt.dat` / `.res.dat` pair), both off by default. Test cleanup code must
therefore treat `.dat` as optional.

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

## Superconvergence

`Superconvergent = true` switches the residual and Jacobian to the interpolatory
HDG_k scheme of Chen, Cockburn, Singler & Zhang (*J Sci Comput* 81:2188): the
physics is evaluated at the `k+2` nodes of the degree-`k+1` basis with the
postprocessed `u*` in place of `u_h`, and interpolated into `P_{k+1}` rather than
`P_k`. The reconstruction itself is `Postprocessing.{hpp,cpp}` and is built for
every run with `k >= 1`, flag or no flag, so `u_star` appears in the netCDF
output and in the `.dat` files unconditionally.

Three tests carry this, in increasing order of scope.

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

**`MMSConvergenceTests.cpp`** measures the observed orders, flag off and flag on,
for `u` and for `u*`:

| case | k | flag off: u, u* | flag on: u, u* |
|---|---|---|---|
| linear, constant kappa | 1 | 1.96, 2.19 | 1.96, **3.05** |
| linear, constant kappa | 2 | 2.97, 4.08 | 2.97, 4.03 |
| nonlinear reaction `u^3 - u` | 1 | 1.96, 2.23 | 1.97, **3.07** |
| nonlinear reaction `u^3 - u` | 2 | 2.95, 4.10 | 2.97, 4.03 |

Read that table carefully, because it is not what a first reading of the papers
predicts. With the flag on, `u*` reaches `k+2` in every case and `u_h` keeps its
optimal `k+1`. With the flag off, `u*` superconverges at `k = 2` but *not* at
`k = 1`, and this is true whether or not the source is nonlinear. So the flag
restores the extra order at `k = 1` and preserves it at `k = 2`.

In particular the loss is not driven by the nonlinearity here, as the papers'
analysis of `I_h F(u_h)` would suggest. Interpolating a *known* smooth source at
the Chebyshev nodes leaves an error that is very nearly `L2`-orthogonal to `P_k`,
so it does not pollute the duality argument the way evaluating `F` at the
`O(h^(k+1))`-accurate `u_h` does. The tests therefore assert what is measured --
`u*` reaches `k+2` with the flag on, and `u_h` does not regress -- and do not
assert that the flag improves on the flag-off rate, because for these problems
there is not always anything to improve.

### What is not covered

* **A general nonlinear flux `sigma_hat(u, q)` is outside the papers' theory.**
  Their conclusion names `F(grad u, u)` as open. `SolveJacTests.cpp` shows the
  Jacobian is right for such a flux, but no order study asserts `k+2` for one.
* **`k = 0`** is rejected (`std::invalid_argument`). The degree-0 `NodalBasis`
  returns from its constructor before building `Vandermonde` or
  `BarycentricWeights` (`Basis.hpp:369-377`), so it cannot be evaluated off-node.
  Paper I requires `k >= 1` for the superconvergence in any case.
* **Spatial adjoint parameters** are rejected with the flag on: they index the
  parameter vector by node, so the star node set would silently redefine how many
  parameters there are.
* **`nAux > 0` and `nScalars > 0`** are handled by the flag-on Jacobian, but no
  order study covers them.
* `dSourcedPhi_Mat` (`Matrices.cpp`) and the second `dAux_Mat` overload build
  their blocks by Gauss quadrature while the residual interpolates, so for
  `nAux > 0` they were *already* inconsistent with the residual before this work.
  The flag-on path uses interpolatory forms; the flag-off path is left exactly as
  it was so regression output stays bit-identical.

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
