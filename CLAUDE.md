# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MaNTA (Maryland Nonlinear Transport Analyzer) solves 1-D reaction–diffusion /
transport systems with a hybridizable discontinuous Galerkin (HDG) spatial
discretisation, integrated in time as an index-1 DAE by SUNDIALS IDA.

`README.md` covers first-time setup (dependencies, `Makefile.local`, installing
SUNDIALS). `Tests/README.md` covers test conventions and the current known gaps
in detail — read it before adding tests or interpreting a coverage number.

## Commands

```sh
make MaNTA                # the solver only (bare `make` also builds python + runs tests)
make test                 # Boost.Test C++ unit tests
make regression_tests     # solver over Tests/RegressionTests/*.conf vs checked-in .ref.nc
make python               # the pybind11 extension, python/MaNTA<suffix>.so
make python_tests         # pytest suite for that extension
make coverage             # rebuild instrumented, run all three suites, write coverage/
make clean                # also sweeps orphaned PhysicsCases/*.o and .d files
```

The regression and Python suites need `requirements.txt` installed and the
virtualenv on `PATH` (the regression driver's shebang is `env python3`):

```sh
export PATH="$PWD/.venv/bin:$PATH"
```

Running one test:

```sh
Tests/UnitTests/UnitTests --run_test=solve_jac_tests/solve_hdg_jac_agrees_with_a_dense_solve --log_level=all
Tests/UnitTests/UnitTests --run_test=mms_convergence_tests --log_level=message   # see BOOST_TEST_MESSAGE output
pytest python/Tests/test_adjoint.py::test_adjoint_gradient_matches_finite_differences
Tests/RegressionTests/TestSolutions.py --tolerance 1e-2
```

All three suites run from any working directory. New unit-test `.cpp` files must
be added to `TEST_SOURCES` in `Tests/UnitTests/Makefile`.

Build variants (set on the make command line, e.g. `make DEBUG=on test`):
`DEBUG` (`-O0 -g -DDEBUG -DPHYSICS_DEBUG`, and `State.hpp`'s `checkShapeAndSet`
becomes shape-checking rather than a plain assignment), `OMP` (enables the
`#pragma omp parallel for` in the batched physics wrappers), `COVERAGE`,
`VERBOSE`, `XLA_FFI`/`CUDA` (JAX FFI, needs jaxlib headers).

## Architecture

### The equation being solved

A physics case defines, per variable `i`:

```
a_i d_t u_i + d_x sigma_i = S_i(u, q, sigma, phi, x, t)
sigma_i    = sigma_hat_i(u, q, x, t)          # the flux
q_i        = d_x u_i                          # introduced as an unknown
G_j(phi, u, q, sigma, x) = 0                  # nAux algebraic auxiliary constraints
G_s(mu, y, dy/dt, t)     = 0                  # nScalars global (non-spatial) unknowns
```

`sigma`, `q`, `u` and the auxiliary variables `phi` live per cell; `lambda` is
the HDG trace unknown on cell faces; `mu` are the global scalars. That ordering —
**`[sigma | q | u | aux]` per cell, then all of `lambda`, then `mu`** — is the
DOF layout of both the solution vector (`DGSoln::Map`) and the local Jacobian
block `MX`, and getting a column index wrong there is the most common way to
break the solver silently. Note that only `PhysicsCases/` may be physics; the
core is generic.

### Solve path

`Solver.cpp:runSolver` is the driver. IDA is handed a **custom
`SUNLinearSolver`** (`SunLinSolWrapper`) plus a **deliberately empty `SUNMatrix`**
(`SunMatrixWrapper`) whose only job is to convince IDA it has a matrix-based
direct solver; the Jacobian is never assembled.

* `SystemSolver::residual` — evaluates the whole residual. Does **not** write the
  Dirichlet boundary rows; those constraints are imposed inside the linear solve,
  so a finite-differenced Jacobian is rank-deficient by exactly the number of
  Dirichlet boundaries.
* `SystemSolver::updateMatricesForJacSolve` — builds and factorises the per-cell
  `MX` blocks from `Matrices.cpp`.
* `SystemSolver::solveHDGJac` — static condensation onto `lambda`, then
  back-substitution.
* `SystemSolver::solveJacEq` — wraps that in a Woodbury/bordered elimination for
  the global scalars.

Because the Jacobian is never formed, **an error in it does not produce a wrong
answer — only slow Newton convergence**. That is why several defects in this area
survived a passing regression suite for months, and why the tests that matter here
are `SolveJacTests.cpp` (finite-difference the residual, require `J dy = g`) and
`MMSConvergenceTests.cpp` (observed order of accuracy).

### Superconvergence (`Superconvergent = true`)

MaNTA *is* an interpolatory HDG method: `residual` evaluates `SigmaFn`, `Sources`
and `AuxG` at the `k+1` nodes of the degree-`k` nodal basis and applies
`InterpolateOntoBasis`, i.e. `I_h F(u_h)` with `I_h` mapping into `W_h = P_k`.
`Matrices.cpp` cites arXiv:1811.09667 for the Jacobian form, which is exactly the
paper that method comes from. Its two sequels (`SuperconvergentHDG-I/II.pdf`)
exist because that scheme loses the `k+2` superconvergence of the postprocessed
solution, and paper I's fix is what the flag implements.

`Postprocessing.{hpp,cpp}` reconstructs `u* ∈ P_{k+1}` per cell from `(u_h, q_h)`
by the local Neumann problem of paper I eq. (7) — sign-flipped, because MaNTA
carries `q = +d_x u` and the paper `q = -grad u`. Eliminating the Lagrange
multiplier gives `gamma = B11 alpha_q + B12 beta_u`, so the per-cell operators are
built once in `initialiseMatrices` and reused. Two more come with them:
`V = [phi_j(x_m)]` samples a `P_k` field at the `k+2` star nodes, and
`A9 = [(chi_m, phi_i)]` projects a `P_{k+1}` interpolant onto the `P_k` test
space, replacing the mass matrix `InterpolateOntoBasis` applies.

With the flag on, the physics is evaluated on the star nodes with `u*` in place of
`u_h`, and every Jacobian block gains a chain factor
(`SystemSolver::accumulateStarBlocks`):

```
d/d(u coeffs)     = A9 diag(dX/du) B12
d/d(q coeffs)     = A9 [ diag(dX/dq) V + diag(dX/du) B11 ]
d/d(sigma, phi)   = A9 diag(dX/dZ) V
```

`diag(dX/du) B11` in the `q` column is the only new coupling: `u*` depends on the
cell's `q` as well as its `u`. Every block stays `(k+1)x(k+1)` and `u*` is
cell-local, so **the DOF layout, `MX`, `solveHDGJac`, `solveJacEq`, the restart
format and the pybind11 casters are all untouched.** `ComputePhysics` loops over
`states.size()`, so no physics case — C++, Python or JAX — needs changing to be
evaluated at `k+2` points instead of `k+1`.

The reconstruction is built for every run with `k >= 1` regardless of the flag, so
`u_star` is always in the netCDF output and the `.dat` files; the flag controls
only whether the *method* uses it. `Tests/README.md` has the measured orders and
the list of what is not covered.

### Non-owning state views

`DGSoln` / `DGApprox` are **`Eigen::Map` views over memory SUNDIALS owns**, not
containers. `SystemSolver::y` maps the `N_Vector` that `runSolver` allocates *and
destroys*, so it dangles after a run; `yJac`/`dydtJac` own their memory
(`yJacMem`) and are what outlives the solve. Anything reading "the solution"
after `runSolver` must use `yJac`.

### Physics cases

`PhysicsCases::map` is a process-global map populated by static-initialisation
side effects: `REGISTER_PHYSICS_HEADER(T)` in the header plus
`REGISTER_PHYSICS_IMPL(T)` in the `.cpp` instantiate a `PhysicsCaseRegister<T>`
whose constructor inserts a factory. Consequences:

* A case only appears if its object file is linked in — nothing references it
  directly, so a missing entry is a link-line problem with no compile error.
* `RegisterPhysicsCase` uses `map::insert`, so a **duplicate name is silently
  ignored and the first registration wins**.
* The map is never reset, so tests must use unique throwaway names.

Two layers sit above `TransportSystem`: cases may implement its virtuals
directly, or derive from `AutodiffTransportSystem` and supply `Flux`/`Source` in
terms of `autodiff` types, which then derives every Jacobian entry automatically.

**Every physics hook exists in two forms**: pointwise (`SigmaFn(i, State, x, t)`)
and batched (`SigmaFn(i, GlobalState, positions, t)`). The batched defaults in
`TransportSystem.hpp` are serial loops over the pointwise version, several under
`#pragma omp parallel for`; a case may override either level.

### Python layer

`Python.cpp` defines the `MaNTA` module. Three pieces to know:

* **Trampolines.** `PyTransportSystem` / `PyAdjointProblem` dispatch each virtual
  to a Python override. `PyTransportSystem::initializeOverrides` probes the
  subclass once and classifies it as *pointwise* or *vectorised* (the latter
  requires both `ComputePhysics` and `ComputePhysicsDerivatives`); it also
  enforces the extra hooks required when `nScalars > 0` or `nAux > 0`. Look up
  overrides with `override_for(name)`, never `method_overrides[name]` — the
  latter default-constructs a null `py::function` and calling it segfaults.
* **Type casters.** `State` ↔ `dict` of 1-D arrays;
  `GlobalState` ↔ `dict` of `(nPoints, nVars)` arrays. **The `GlobalState` caster
  transposes in both directions** (C++ stores `(nVars, nPoints)`), so a
  round-trip test cannot detect a missing transpose on its own — check the
  orientation from inside a batched call instead.
* **`PyRunner`** (`configure(dict)` / `run` / `run_ss` / `getSolution` /
  `getAdjointGradients`) is the API the optimisation drivers use, and the only
  route supporting repeated configure/run cycles in one process. Its parameter
  table is declarative and lives at the top of `PyRunner.cpp`.

JAX physics cases (`python/JAXTransportSystem.py`, `python/State.py`) wrap the
dict interface in equinox modules via the `MaNTA_Decorator` / `Physics_Decorator`
adapters.

### Adjoints

`AdjointProblem` defines an objective `G = ∫ g dx` and its parameter
derivatives; `SystemSolver::initializeMatricesForAdjointSolve` →
`solveAdjointState` → `computeAdjointGradients` fill `G_p`.
`np` splits into `np_boundary` trailing boundary parameters and the rest
internal; several defaults loop only to `getNpInternal()`.

## Traps worth knowing before you edit

* **autodiff expression templates hold references to their operands.** A lambda
  passed to `trapezoid` (or anything that stores the result) must declare
  `-> Real` explicitly; with a deduced return type it hands back an expression
  referring to dead temporaries, and the symptom is a silently wrong (often zero)
  answer rather than a crash. See `Tests/UnitTests/UtilityTests.cpp`.
* **Never slice an Eigen `solve()` result.** `lu.solve(B)` returns a lazy `Solve<>`
  expression with no coefficient accessor, so `lu.solve(B).topRows(n)` compiles and
  then corrupts the heap. It cost an afternoon in `Postprocessing.cpp`, where the
  symptom was a SIGSEGV inside `free()` in an unrelated static's destructor.
  Assign to a `Matrix` first, then slice.
* **Include `<Eigen/Core>` and `<Eigen/Dense>` before the project headers**, the
  way `SystemSolver.hpp` does. The build defines `EIGEN_USE_BLAS`, which swaps in
  BLAS-backed product specialisations; a header that reaches Eigen only through
  `Basis.hpp`'s `<Eigen/LU>` gives its translation units a different set of
  definitions from ones that include `<Eigen/Dense>`. LTO then picks one, and the
  symptom is heap corruption surfacing at exit in whichever static destructor runs
  first — for us, `ChebyshevBasis::singletons`, which the change had nothing to do
  with. `Postprocessing.hpp` carries a comment saying so.
* **Build staleness has bitten three times.** Header dependencies now come from
  `-MMD -MP`, and the `python` target lists `MaNTA.o`, `Python.cpp` and
  `PyRunner.cpp` as prerequisites. If a fix appears to have no effect, check the
  object timestamps before doubting the fix.
* **The top-level Makefile has a bare `export`.** A recursive `$(MAKE)` inherits
  the already-computed release `CXXFLAGS`, which is why the `coverage` target
  runs `env -u CXXFLAGS -u LDFLAGS $(MAKE) COVERAGE=on`.
* **Output filenames come from the config file's *stem*** (`Solver.cpp` uses
  `inputFilePath.stem()`), so `.nc` / `.dat` / `.restart.nc` land in the current
  directory regardless of any path in `OutputFilename`.
* **`printSources` reads the source cache through a basis of the residual's
  order.** With `Superconvergent = true` the cache holds `k+2` values per cell
  rather than `k+1`, so `SystemSolver::print` picks its basis and stride from the
  flag; hardcoding `k+1` there reads across cell boundaries.
* **netCDF is the default output; the `.dat` files are opt-in.** A run writes
  `<stem>.nc` and `<stem>.restart.nc` unconditionally. The plain-text gnuplot
  output needs `WriteDatFile = true`, and `<stem>.dydt.dat` / `<stem>.res.dat`
  need `WriteDebugDatFiles = true` *and* a `PHYSICS_DEBUG` build. Both options
  default to false and are accepted by `runManta` and `PyRunner::configure`
  alike.
* **Tests reach private `SystemSolver` members** through `MANTA_TEST_PRIVATE`,
  which a `-DTEST` build widens to `public`. No friend declarations needed.
* **gcov counts a templated line once per instantiation**, which makes
  `NetCDFIO.hpp` and `util/trapezoid.hpp` look far worse than they are. Judge
  those by distinct uncovered lines; `Tests/README.md` has the numbers.

## Known limitations

These are deliberate and documented, not oversights — see `Tests/README.md` and
`TODO` for the full versions:

* `WriteAdjoints()` is commented out at `Solver.cpp:350`, so no run serialises
  adjoint output. The gradients themselves are verified through
  `PyRunner::getAdjointGradients` in `python/Tests/test_adjoint.py`.
* Restarting is fragile at tight tolerances, more so with `nAux > 0`; each
  regression round-trip case runs at the tightest tolerance that completes.
* `python/Tests/test_reference_solutions.py::test_jax_aux_test` is a `strict=True`
  xfail. The C++ `nAux > 0` path is known good (`python/Tests/test_aux.py`), so
  the fault is in the JAX fixture or `JAXTransportSystem`'s aux hooks.
* `PhysicsCases/CurvedMirrorPlasma/` has never compiled and is excluded from
  `PHYSICS_SOURCES`.
* The `UseMMS` options on `LinearDiffusion` and `LinearDiffSourceTest` are not
  usable as-is (the manufactured solution does not satisfy the boundary
  conditions; the latter never applies `MMS_Source` at all).
