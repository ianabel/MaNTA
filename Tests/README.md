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

  Three more cases have landed since, each also provoked by a defect:

  * `the_SER_rate_and_floor_change_the_cost_and_not_the_answer` measures the schedule through physics evaluations -- 552 at the defaults, 3540 with the floor at 1, 1704 with the floor at 1 and the rate at 2 -- and requires the converged state to be identical in all three. An option that changed the answer would be a bug; one that changed nothing would be inert.
  * `the_steady_diagnostics_count_the_whole_solve_not_the_last_step`. **KINSOL zeroes its own counters at the top of every `KINSol` call**, so the continuation loop has to sum them as it goes; reading them once at the end -- the obvious thing, and what this did first -- reported 1 Newton iteration against 5 continuation steps and 35 Jacobian solves. Self-evidently impossible, and it still looks like a number, which is why the test asserts invariants (`newtonIters >= steps`, `residualEvals == kinFuncEvals + steps + 1`) rather than values. The second of those also pins the counter snapshot being taken before the first `steadyNorm()`, which it was not to begin with.
  * `a_failed_steady_solve_still_writes_the_last_state_it_reached`, using a tolerance nothing can reach so the solve stalls at ~1e-16 and exits by the "ran out of continuation steps" path.

  What is still uncovered is the rest of the algorithm -- step rejection,
  `Newton` mode, the `KINSetMaxNewtonStep` clamp, and the hard-`KINSol`-failure
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
