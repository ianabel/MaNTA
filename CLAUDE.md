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

**The second line above is a sign convention, not an identity: the stored `sigma`
is `-sigma_hat`.** `residual` forms the flux row as
`res.sigma = A sigma_h + (I_h sigma_hat, phi)` with `A` the mass matrix, so what
it enforces is `sigma_h = -Pi(sigma_hat)`. (`setInitialConditions` does it
explicitly, with a "remember minus sign" comment.) The PDE actually integrated is
therefore

```
a_i d_t u_i - d_x[ sigma_hat_i(u, q, x, t) ] = S_i
```

Two consequences. A manufactured source must be differentiated with that minus
sign — the check is `ManufacturedDiffusion` in `MMSConvergenceTests.cpp`, whose
`SigmaFn` returns `kappa q` against `S = sin(pi x)(1 + kappa pi^2 (1+t))`, which
is `u_t - kappa u_xx` for `u = sin(pi x)(1+t)`: a diffusion equation, not the
anti-diffusion `+` would give. Get it backwards and the case still converges, to
the wrong function, at the right rate — so an order study will not catch it, only
a closed-form comparison will. And `State::Flux[i]`, which physics hooks read,
carries the negated `sigma_h`, not `sigma_hat`.

### Solve path

`Solver.cpp:runSolver` is the driver. It is three phases, which can also be
called separately:

* `SystemSolver::initialize` — allocate the SUNDIALS objects, build the initial
  condition, open the output files, run `IDACalcIC`.
* `SystemSolver::integrate(tFinal)` — the time loop, then the adjoint solve and
  the final netCDF / restart output.
* `SystemSolver::destroySundials` — free all of it. Idempotent, and safe with no
  preceding `initialize`, which is what lets `runSolver` free on both the normal
  and the exceptional path.

Every SUNDIALS handle is a member, not a local, so those three can be split.
`ctx` is the exception: it belongs to the `SystemSolver`, not to a run, and
`destroySundials` must not touch it.

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
containers. `SystemSolver::y` maps the `N_Vector` that `initialize` allocates and
`destroySundials` frees, so it dangles after a run; `yJac`/`dydtJac` own their
memory (`yJacMem`) and are what outlives the solve. Anything reading "the
solution" after a run must use `yJac`. `initialize` seeds it with the initial
condition, so it is also valid *before* `integrate` — which matters now that a
caller can stop between the two.

`DGSolnImpl` holds its basis **by value** (`const BasisType Basis`), so
`getBasis()` returns a reference into the `DGSoln`, not into a shared singleton.
Binding that to a reference which outlives the owning object is a use-after-free;
`SystemSolverTests.cpp` did it for years and only started failing when an
unrelated new test file changed the allocation pattern.

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
* **`PyRunner`** (`configure(dict)` / `run` / `run_ss` / `getSolution` / `G` /
  `getAdjointGradients`) is the API the optimisation drivers use, and the only
  route supporting repeated configure/run cycles in one process — it works by
  building a *fresh* `SystemSolver` in every `configure()` (`PyRunner.cpp:117`),
  which is load-bearing; see Known limitations. Its parameter table is
  declarative and lives at the top of `PyRunner.cpp`.

  `G` returns the objective without the gradient. The saving is in the run, not
  in `G` itself: `integrate` calls `runAdjointSolve()` whenever `solveAdjoint` is
  set, so with `solveAdjoint = True` the gradients are already computed by the
  time `run` returns and `getAdjointGradients` only reads `G_p`. Configure with
  `solveAdjoint = False` to skip the adjoint solve, and `G` builds an
  `AdjointProblem` on demand purely to evaluate `GFn`. That object is kept in a
  separate member from `adjoint` so its presence can never be mistaken for "the
  gradients have been computed".

**The scalar hooks' Python signatures are not the C++ ones.** `ScalarGExtended`
and friends take `DGSoln` and `Interval`, which have no Python representation, so
`PyTransportSystem` evaluates on the nodes and passes a `GlobalState` plus the
quadrature data instead. What Python must implement:

```
InitialScalarValue(s)                                    -> float
InitialScalarDerivative(s, states, states_dot, weights)  -> float
isScalarDifferential(i)                                  -> bool
ScalarG(s, states, states_dot, weights, t)               -> float
ScalarGPrime(states, states_dot, weights, phi_boundary, t)
    -> (list of nScalars GlobalState dicts,   d G_s / d state
        list of nScalars GlobalState dicts)   d G_s / d state_dot
dSources_dScalars(s, state, x, t)  -> vector of length nScalars
```

`weights` is one quadrature weight per node (`nCells*(k+1)`), so an integral over
the domain is `weights @ u`; `phi_boundary` is `(k+1, 2)`. Note
`dSources_dScalars` is indexed by *scalar*, not by variable, and that
`InitialScalarDerivative` is only consulted for scalars where
`isScalarDifferential` is true. `python/Tests/test_scalars.py` exercises all of
it against a closed form, in both the algebraic and differential cases.

JAX physics cases (`python/JAXTransportSystem.py`, `python/State.py`) wrap the
dict interface in equinox modules via the `MaNTA_Decorator` / `Physics_Decorator`
adapters.

### Adjoints

`AdjointProblem` defines an objective `G = ∫ g dx` and its parameter
derivatives; `SystemSolver::initializeMatricesForAdjointSolve` →
`solveAdjointState` → `computeAdjointGradients` fill `G_p`.
`np` splits into `np_boundary` trailing boundary parameters and the rest
internal; several defaults loop only to `getNpInternal()`.

`initializeMatricesForAdjointSolve` assembles the same local blocks as
`updateMatricesForJacSolve` and stores `M.transpose()`, so **the two must be kept
in step block for block** — the adjoint operator *is* that transpose. This is
where the two functions differ in consequence: a block missing from the forward
Jacobian only slows Newton down, but a block missing from the adjoint matrix
produces a silently wrong gradient with a perfectly good `G`. The `dSigma/dPhi`
block was absent here for exactly that reason, and cost nothing visible until
`python/Tests/test_adjoint_aux.py` was written. If you add a coupling to one
function, add it to the other.

Anything indexed per auxiliary variable is sized `nAux`, not `nVars`. Those
coincide in every fixture except `test_adjoint_aux.py`, so a confusion between
them is invisible in the rest of the suite; `dGdaux_Vec` had two.

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
* **`-Wno-parentheses` is global, and on gcc it takes `-Wdangling-else` with it.**
  So gcc will not tell you about a dangling `else`; clang will, because it treats
  `-Wdangling-else` as a separate warning. Build with clang occasionally — that is
  what CI's second job is for. The same applies in reverse: gcc never diagnoses a
  polymorphic base with a non-virtual destructor, clang does
  (`-Wdelete-non-abstract-non-virtual-dtor`), and it reports it at the point of
  *destruction* inside libstdc++, once per instantiating translation unit, which
  makes the message look like a standard-library problem rather than yours.
* **Third-party includes use `-isystem`, not `-I`** (`Makefile.config`: SUNDIALS,
  toml11, Boost, netCDF, Eigen, autodiff, and pybind11 in the top-level Makefile).
  `-Werror` is on, and Eigen's own headers do trip `-Wunused-but-set-variable`
  under clang — reachable only from the pybind11 build, which pulls in
  `SparseCore`. Adding a dependency with `-I` re-arms that.
* **`COMPILER_ID` in `Makefile.config`** distinguishes gcc from clang for the few
  flags that differ: `-fprofile-abs-path` (gcc-only, a hard error on clang),
  `-fno-inline-small-functions` / `-fno-default-inline` (gcc-only, ignored with a
  warning by clang), and `GCOV`, which is `gcov-14` for gcc but has to be
  `llvm-cov gcov` for clang. `-flto=auto` is fine on both.
* **`GCOV` is derived outside `ifdef COVERAGE`, deliberately.** The `coverage`
  target runs `gcovr` in the *parent* make and only recurses with `COVERAGE=on`,
  so the make that expands `$(GCOV)` is the one where `COVERAGE` is undefined.
  Deriving it inside that branch left the parent with a bare `gcov` — which is
  gcov-14 on a box whose default compiler is gcc-14, and gcov-13 on
  ubuntu-24.04, where the image ships gcc 12/13/14 with 13 as the default and the
  workflow builds with `g++-14`. gcov then exits 3 with
  `AdjointVectors.gcno:version 'B42*', prefer 'B33*'`, gcovr promotes that to a
  hard error, and `make coverage` fails with exit 64 on CI while passing locally.
  Anything that reads `.gcno`/`.gcda` must come from the same toolchain version
  that wrote them.
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
  `PyRunner::getAdjointGradients` in `python/Tests/test_adjoint.py` and
  `test_adjoint_aux.py`.
* **A second *integration* on the same `SystemSolver` does not work.** `IDASolve`
  fails with `IDA_ERR_FAIL` (-3) on the first step of the second run. This is not
  a consequence of the three-phase split — `main` before it failed identically
  through two `runSolver` calls — it has simply never been exercised, because
  `PyRunner::configure` builds a fresh `SystemSolver` and the standalone binary
  runs once and exits. Calling `initialize` again after `destroySundials` *does*
  work, and rebuilds the initial condition; it is completing a second time loop
  that fails. Undiagnosed; `Tests/README.md` has the detail.
* Restarting is fragile at tight tolerances, more so with `nAux > 0`; each
  regression round-trip case runs at the tightest tolerance that completes.
* `python/Tests/test_reference_solutions.py::test_jax_aux_test` is a `strict=True`
  xfail. The C++ `nAux > 0` path is known good (`python/Tests/test_aux.py`), so
  the fault is in the JAX fixture or `JAXTransportSystem`'s aux hooks.
* `PhysicsCases/CurvedMirrorPlasma/` has never compiled and is excluded from
  `PHYSICS_SOURCES`.
* The `UseMMS` options on `LinearDiffusion` and `LinearDiffSourceTest` have been
  removed: the first's manufactured solution did not satisfy its own boundary
  conditions, and the second never applied `MMS_Source` at all. `MirrorPlasma`
  still implements `MMS_Solution` against `AutodiffTransportSystem`'s facility,
  which is deliberate and untouched. Order of accuracy is measured by
  `Tests/UnitTests/MMSConvergenceTests.cpp`, which builds its own manufactured
  problems.
