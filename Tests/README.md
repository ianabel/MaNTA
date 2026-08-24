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

**All three round trips now survive 1e-6 / 1e-8** -- see the measured table
beside the calls in `TestSolutions.py`. They did not: that table used to read
`IDASolve -4` for `AuxVarTest` at 1e-4 and tighter, and `-6` for `MatTest` at
1e-6, and this paragraph used to say restarting was fragile at tight tolerances.
Two fixes closed it, **neither of them in the restart machinery**:
`setInitialConditions` finishing every restart with `EvaluateLambda()` and so
discarding the converged trace the file carried, and `AuxVarTest`'s missing
`dSigma_dPhi` block (both described below).

What is left is a ceiling belonging to the *cases*, at 1e-8 / 1e-10, where
`MatTest`'s **uninterrupted** run fails as well -- so the restart is not
implicated there -- and the other two resume-fail with `IDA_ERR_FAIL` (-3)
rather than the corrector failure (-4) that used to appear. Only `AuxVarTest`
was tightened to 1e-6 along with `LinearDiffusion`; `MatTest` stays at 1e-4
because 1e-6 costs 101 s against 6.0 s for agreement of 2.7e-10 that nothing
needs.

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
  missing-file path is tested. A plugin needs `make install` and a shared object
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

* **The dG/dt gate needs a consistent initial condition, and now forces one.**
  `an_armed_gate_is_given_a_consistent_initial_condition` in
  `AlgebraicDerivativeTests.cpp`. Once `initialize()` gained two ways to skip
  `IDACalcIC` -- a steady solve, and a restart that was copied rather than
  projected -- the gate started differentiating the guess, and on
  the guess `dG/dt` comes out with the wrong *sign*: `AuxDiffusion` with
  `G = Int u dx` gives +1.654 corrected against -1.769, `ScalarDiffusion` +2.208
  against -1.187. The gate rejects on `dGdt < -tol`, so it abandoned runs it
  should accept.

  The test asserts both halves at both fixtures: that an armed gate on a steady
  solve does run `IDACalcIC`, that the same solve *without* the gate does not, and
  that the armed value equals the time-marching one to 1e-12. Its guard is that
  the armed and unarmed values straddle zero -- so a fixture that stopped
  distinguishing a fixed gate from a broken one fails rather than passes.
  `TestDiffusion`, which every older gate test uses, agrees to 3.6e-16 either way
  and is exactly why this went unnoticed.

  `a_failed_calcic_that_only_the_gate_wanted_leaves_the_run_alone` in
  `SolverLifecycleTests.cpp` covers the other half. Forcing `IDACalcIC` puts it on
  the states it is likeliest to fail on, so it fails open: the run continues from
  the guess, `IDAReInit` restores IDA's `phi` so the fallback is bit for bit the
  unarmed run, and the gate declines instead of answering. Both of its premises
  are `BOOST_TEST_REQUIRE`, so if the run ever stops skipping `IDACalcIC` on its
  own, or `IDACalcIC` starts succeeding there, the case says so rather than
  passing for the wrong reason -- and **that is how the `tout1` bug above was
  found**: fixing it turned the first guard red, which is what a guard is for.

  Its provocation was rewritten as a result, and the rewrite is worth reading
  before touching it. It is now a *steady* solve on `MisdeclaredScalar`, whose
  differential scalar has no time derivative in its `ScalarG`: `IDA_YA_YDP_INIT`
  freezes every differential value, so that row is a constant no Newton direction
  can touch and `IDACalcIC` exhausts its linesearch. Both halves of that are
  load-bearing.

  * **The failure has to be a property of the problem**, not of MaNTA, or the
    test pins a bug. A declaration error is the one class that qualifies -- it is
    what kept `python-physics/mirror-plasma`'s voltage controller from ever
    starting -- and the scalar's initial value has to sit in a window: consistent,
    and the Newton step is zero so `IDACalcIC` *converges* (irreducible only beats
    small when there is something to reduce); far out, and nothing downstream
    survives either.
  * **The run has to be a steady solve.** A time-marching run that cannot complete
    `IDACalcIC` usually cannot integrate either -- the differential values it
    failed to reconcile are in IDA's local error test, and the jump to their
    consistent values does not shrink with `h`, which is `IDA_ERR_FAIL` on the
    first step. So "CalcIC failed but the run was fine" is a far narrower claim
    than it sounds, and a steady solve, where KINSOL drives the whole residual to
    zero from wherever it starts, is most of what is left of it.
    `python-examples/jardin-critical-gradient` is the same shape.

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
  `make python_tests` (`pytest.ini` is `testpaths = python/Tests`) and needs
  `desc` and `optimistix`. That is a real reduction in what CI covers, recorded
  here rather than left to be discovered.

  `CurvedMirrorPlasma/` had never compiled in any case (commit `c17fa42`,
  "start to add in curved stuff (doesn't compile)"): 49 errors, including
  references to a `CurvedMagneticField` class and a `PlasmaTypes` enum that were
  never written. It was excluded from `PHYSICS_SOURCES` for that reason, and it
  depended on `MirrorPlasma` and `PlasmaConstants`, so it could not have
  outlived them.

* **`PhysicsCases/` is reported but not gated.** It is exercised as test
  fixtures rather than as a coverage target in its own right.
