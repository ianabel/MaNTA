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
make python               # the manta package, python/manta/_manta<suffix>.so
make install PREFIX=...   # headers under include/manta, libmanta.so, manta.pc
pip install .             # the `manta` package and the `manta` console script
pip install .[jax]        # ...and manta.jax (jax, equinox, jaxtyping)
make python_tests         # pytest suite for that extension
make coverage             # rebuild instrumented, run all three suites, write coverage/
make docs                 # docs/_build/html, via .venv-docs built from
                          # docs/requirements.txt; -W, as Read the Docs runs it
make stubs                # regenerate python/manta/_manta.pyi from the module
make stubs-check          # fail if the committed stub is stale (CI runs this)
make typecheck            # mypy over the manta package
make clean                # also sweeps orphaned PhysicsCases/*.o and .d files,
                          # python/manta modules of every ABI suffix, the
                          # bytecode and pytest caches, and clean_data below
make clean_data           # run output (.nc/.restart.nc/.dat) at the root and in
                          # Tests/RegressionTests, python/Tests/ and each
                          # directory under python-examples/ and python-physics/

./MaNTA --list-options    # every configuration key, straight from ConfigSchema.cpp
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
a closed-form comparison will. And the stored flux a hook reads
carries the negated `sigma_h`: `State::sigma(i)` is that stored value,
`State::sigmaHat(i)` the physical flux `SigmaFn` returned. Nothing in the tree
reads the incoming sigma at all — `AdjointPlasma` threads it into
`Sn`/`Spi`/`Spe` without using it, and every `dSources_dsigma` is
zero — so a case that starts using it is the first to care, and no test would
catch a sign error there.

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
`RF_cellwise` trap below).

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

**When IDA does give up, get its own account of the step before theorising.**
SUNDIALS is built here with `SUNDIALS_LOGGING_LEVEL 4`, and `SUNContext_Create`
builds its logger from the environment (`sundials_context.c` calls
`SUNLogger_CreateFromEnv`), so

```sh
SUNLOGGER_INFO_FILENAME=/tmp/ida-info.log SUNLOGGER_DEBUG_FILENAME=/tmp/ida-debug.log ./MaNTA foo.conf
```

yields IDA's per-attempt record — step size, order, `dsm`, and whether the attempt
died in the error test or the nonlinear solve — for no code change at all. Write it
to a file rather than to `stderr` if a test is running, because `CapturedOutput`
redirects the standard descriptors.

Two return codes worth being able to read without looking them up:

* **`IDA_ERR_FAIL` (-3) on the *first* step means the initial condition violates
  the algebraic constraints.** It takes ten error-test failures in one step attempt
  to produce it, and each one cuts `h`; a local error estimate that will not shrink
  with `h` is the signature of an inconsistent state, because `IDASetSuppressAlg`
  is never called and so `sigma`, `q`, `lambda` and `phi` are all in the error
  test. Look at what `IDACalcIC` was given, not at the time loop.
* **`IDA_CONV_FAIL` (-4) from `IDACalcIC`** is the same problem one stage earlier:
  the Newton/linesearch could not reach a consistent state from the guess
  `setInitialConditions` built. The guess is worth suspecting before the solver is.
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
  unknowns each failing row can actually reach**: print `ScalarGPrime`'s `dGdot`
  and check it is nonzero for every differential scalar. The trap is that the
  residual can be tiny — 4.3e-6 there, just the difference between two quadrature
  rules for the same integral — and still fatal, because irreducible beats small.

### Configuration

**Every key MaNTA accepts is declared once**, in `ConfigSchema.cpp`: canonical
name, deprecated aliases, type, category, per-reader requiredness, default and a
line of documentation. `./MaNTA --list-options` prints the table. There used to
be two — `runManta` open-coded ~120 lines of `toml::find_or`, `PyRunner::configure`
carried its own `params` list — and they had drifted: two names for the initial
time (`t_initial`/`tZero`), two defaults for `Absolute_tolerance` (`1e-2` against
`1e-3`), and six keys that existed on one surface only. Nothing reported any of
it, and `docs/configuration.rst` carried a hand-maintained table of the
differences.

The path is `ConfigSource` -> `loadSolverConfig` -> `SolverConfig` -> `makeGrid`
and `applySolverConfig`. `ConfigSource` is the only piece that differs between
the surfaces: `TomlConfigSource` (in `SolverConfig.cpp`) and `DictConfigSource`
(in `PyConfigSource.hpp`). What to know before editing any of it:

* **`ConfigSchema.hpp`, `SolverConfig.hpp` and their `.cpp` files must stay
  pybind11-free.** They link into `MaNTA`, `libmanta.so` and `Tests/UnitTests`.
  `DictConfigSource` is the python-side half and lives in a header `PyRunner.cpp`
  alone includes. Neither header is in `INSTALL_HEADERS`, and neither belongs
  there: an out-of-tree case is handed the parsed `toml::value` and reads its own
  table through toml11, never through the solver's schema.
* **`applySolverConfig` is the single point at which a configuration reaches the
  solver.** That is what stops the two surfaces configuring differently, and it
  is also why a `set*` call dropped from it un-configures *both* at once, where
  the same slip used to cost one. `both_sources_produce_the_same_solver_config`
  compares `SolverConfig`s, not solvers, so it would not notice.
* **`Category` and `Reader` are two different enums.** `Category` says what a key
  *is* (`Solver`, `ProblemSelection`, `Cli`); `Reader` says who is asking
  (`Toml`, `Dict`). Requiredness differs by reader for exactly three keys:
  `t_final` and `TransportSystem` are required of a config file only,
  `OutputFilename` of a dict only — a file falls back to its own stem.
* **Presence, not value, is the signal for `t_final` and `SteadyStateTolerance`**,
  which are `std::optional` in `SolverConfig`. A present `SteadyStateTolerance`
  arms steady-state termination on both surfaces; `run_ss()` arms it regardless,
  falling back to `1e-3`. `Runner.run()` with no argument uses `t_final`, and
  `run(tFinal)` overrides it.
* **An unknown key is an error, with the nearest schema entry suggested.** The
  sweep sees `[configuration]` only — a physics case's own table is read by the
  case and checked against nothing. The three `PythonModule*` keys are in the
  schema as `Category::Cli` purely so the eight config files carrying them are
  not rejected; the solver never reads them. Conversely `TransportSystem` and
  `PhysicsPlugins` are `Category::ProblemSelection` and are an *error* in a dict,
  rather than being accepted and ignored.
* **Configuration errors propagate.** `loadSolverConfig` throws
  `std::invalid_argument`, which pybind maps to `ValueError` for `manta.run()`;
  `main()` catches it and prints one line for the command line.
  `PyRunner::configure` re-throws it as `std::runtime_error` so that surface goes
  on raising `RuntimeError`, which is what it always has. "Could not start"
  conditions — no such config file, an unknown `TransportSystem`, an unopenable
  restart file — still make `runManta` log and return 1.
* **The naming style is deliberately not unified.** `delta_t`, `MinStepSize` and
  `solveAdjoint` keep their inconsistent spellings; only the two genuine name
  *conflicts* were resolved, because regularising the rest would churn every
  config file in the tree for no functional gain.

### Superconvergence (`Superconvergent = true`)

MaNTA *is* an interpolatory HDG method: `residual` evaluates `SigmaFn`, `Sources`
and `AuxG` at the `k+1` nodes of the degree-`k` nodal basis and applies
`InterpolateOntoBasis`, i.e. `I_h F(u_h)` with `I_h` mapping into `W_h = P_k`.
`Matrices.cpp` cites arXiv:1811.09667 for the Jacobian form, which is exactly the
paper that method comes from. Its two sequels (`refs/SuperconvergentHDG-I/II.pdf`, indexed with their dois in
`refs/Refs.md`; the PDFs are gitignored, so fetch them if you need them)
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
  directly, so a missing entry is a link-line problem with no compile error. An
  unknown name now throws from `InstantiateProblem` with the list of what *is*
  registered, rather than returning `nullptr` for every caller to check.
* A **duplicate name throws**. It used to be a bare `map::insert`, so the first
  registration silently won and the second case was unreachable.
* The map is never reset, so tests must use unique throwaway names.
* A case can live **outside this tree**: build it as a shared object against the
  installed headers and name it in the config's `PhysicsPlugins` array, which
  `runManta` dlopens before instantiating. See "Out-of-tree builds" below.

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
anti-dissipative and the run diverges. This replaced
assigning `nVars` in a constructor body, nine naming virtuals,
`isLower/isUpperBoundaryDirichlet`, `isScalarDifferential`, and a pair of
uninitialised bools the boundary virtuals read. A case whose shape depends on its
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

### Python layer

`Python.cpp` defines `_manta`, the compiled core of the **`manta` package**;
`python/manta/__init__.py` re-exports it and adds the parts better written in
Python. Users `import manta`, and `pip install .` makes that work from anywhere
(the build shells out to `make python`, so it still needs `Makefile.local`).
A Python case declares itself with class attributes:

```python
class MyCase(manta.TransportSystem):
    variables = [manta.Field("n", "density", "m^-3", lower=manta.Neumann)]
    def __init__(self, config, grid):
        super().__init__()          # reads the class attributes
```

`super().__init__(spec)` and `super().__init__(variables=[...])` also work, for a
case whose shape depends on its config. `nVars`/`nScalars`/`nAux` are read-only.
**Only `SigmaFn` and `Sources` are required**: an absent derivative hook means
that block is identically zero, which is what the zeroed out-parameter already
gives. Four pieces to know:

* **Trampolines.** `PyTransportSystem` / `PyAdjointProblem` dispatch each virtual
  to a Python override. `PyTransportSystem::initializeOverrides` probes the
  subclass once and classifies it as *pointwise* or *vectorised* (the latter
  requires both `ComputePhysics` and `ComputePhysicsDerivatives`); it also
  enforces the extra hooks required when `nScalars > 0` or `nAux > 0`. Look up
  overrides with `override_for(name)`, never `method_overrides[name]` — the
  latter default-constructs a null `py::function` and calling it segfaults.
* **State is a view, GlobalState is a dict.** A pointwise `State` reaches Python
  as a non-owning window onto solver memory (`PyState.hpp`) with named fields —
  `s.u`, `s.q`, `s.sigma`, `s.sigmaHat`, `s.phi`, `s.scalars` — each indexable by
  position or by declared name. It is valid only inside the call; `np.array(v,
  copy=True)` to keep anything, and note that `__array__` has to honour `copy`
  or numpy hands back a view of a destroyed temporary. There is no way to build
  one from Python, which is why the tests drive the pointwise path through the
  batched entry point. `GlobalState` still crosses as a `dict` of
  `(nPoints, nVars)` arrays — what the JAX path wants — and **its caster
  transposes in both directions** (C++ stores `(nVars, nPoints)`), so a
  round-trip test cannot detect a missing transpose; check the orientation from
  inside a batched call instead.
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

**The scalar hooks are one name each, and the same in both languages.** There
used to be four names and five signatures — `ScalarG`, `ScalarGExtended`,
`ScalarGPrime` and two `ScalarGPrimeExtended` overloads, one taking a
`std::function` test function and an `Interval` and asking the case to integrate
against them — plus a *different*, flatter set on the Python side that the
trampoline translated to. The flat one won:

```
ScalarG(s, y, ydot, abscissae, weights, phiBoundary, t)          -> Value
ScalarGPrime(dG, dGdot, y, ydot, abscissae, weights, phiBoundary, t)
InitialScalarValue(s)                                            -> Value
InitialScalarDerivative(s, y, dydt)   [C++ takes DGSoln; Python takes nodal data]
dSources_dScalars(s, out, state, x, t)   — indexed by *scalar*, not by variable
```

`y`/`ydot` are the solution sampled on the element nodes; `weights` is one
quadrature weight per node, so `Int u dx` is `ScalarHooks::integrate(...)` and
its derivative with respect to node j is `weights(j)`. **A case must use these
weights rather than a rule of its own** — `ScalarTestLD3` used a global adaptive
Kronrod rule over a piecewise polynomial, which is not a smooth function of the
coefficients, and disagreed with its own Jacobian by 8%. `phiBoundary` is
`(k+1, 2)` and is the *only* way to reach a boundary point value, because the
nodes are Chebyshev points of the first kind and strictly interior;
`ScalarHooks::boundaryValue` / `addBoundaryDerivative` do that contraction.
`ScalarGPrime` reports every scalar at once, as d/d(DOF), into buffers that
arrive zeroed.

`checkScalarDerivative` in `ScalarJacobianTests.cpp` finite-differences a case's
own `ScalarG` against its `ScalarGPrime` and is the first thing to run when a
scalar system converges slowly. `python/Tests/test_scalars.py` exercises the
Python side against a closed form, algebraic and differential.

* **`manta.cli`** is the `manta` console script: it reads the config, imports the
  physics module it names, and hands over to the same `runManta` the binary uses.
  That is what lets a Python case and its driver live outside this tree.
  `load_physics_modules` accepts **two forms**, and `python/Tests/util.py`
  delegates to it so the rule has one implementation:

  * `PythonModule` — an importable dotted name, the documented form.
  * `PythonModuleFile` (+ optional `PythonModuleName`) — a path, **resolved
    relative to the config file, not the cwd**. Every config in the tree is
    written as though that were true; the retired `PyManta` resolved against the
    cwd, which is why they only ran from one directory.

  Either way the module is imported for its registrations, and if it defines
  `registerTransportSystems()` that is called afterwards. Registering at import
  via `registerPhysicsCase` is the documented convention; the hook is what every
  example under `python-examples/` actually uses, and without honouring it none
  of those configs run. `python-physics/` is neither: both systems there build a
  `manta.Runner` in Python and register nothing.

JAX physics cases use **`manta.jax`** (`python/manta/jax/`), which wraps the dict
interface in equinox modules via the `MaNTA_Decorator` / `Physics_Decorator`
adapters. It was six flat modules in `python/` until the examples split, and
three properties of the subpackage are load-bearing rather than stylistic:

* **`manta/__init__.py` must never import `.jax`.** JAX is an optional extra
  (`pip install manta[jax]`), so a top-level re-export would make it mandatory
  for every user of the solver — and it would turn the subpackage's relative
  imports into a cycle, since `transport_system.py` does `import manta`.
  `test_jax_layer.py` AST-scans `__init__.py` for exactly this.
* **`manta/jax/__init__.py` must not import `ffi_runner` eagerly.** That module
  registers XLA FFI targets at module scope, and the bindings it looks for —
  `runner_ffi_ops`, `runner_ffi_ops_cuda` — are `#ifdef XLA_FFI`
  (`Python.cpp:361`). Eager, it would take `from manta.jax import State` down on
  every default build. A module-level `__getattr__` serves `FFIRunner`,
  `Platform` and the two `register_ffi_*` helpers instead, and the module itself
  raises an `ImportError` naming the flag rather than an `AttributeError` from
  inside a loop.
* **No module in the layer may write `os.environ` at import.** `FFIRunner.py`
  set three variables and `JAXTransportSystem.py` forced
  `JAX_PLATFORM_NAME=cpu`; as library code that last one would have disabled the
  GPU path `ffi_runner` exists to provide. Those writes live in the drivers that
  need them now — `python-physics/stellarator/`.

`python/` holds exactly two things: `manta/` and `Tests/`. Every driver, config
and notebook is one self-contained directory under either **`python-examples/`**
or **`python-physics/`**, and the split between those two is *purpose*, not
mechanism. Both import `manta` and `manta.jax` absolutely, the way out-of-tree
code does; both survive being copied elsewhere. An example is written to be
read — small, one idea each, no dependency outside `requirements.txt`. A system
under `python-physics/` is run to get physics: `mirror-plasma/` (a package, its
own tests, needs `optimistix`) and `stellarator/` (DESC, yancc, interpax, and an
`XLA_FFI` build). Neither `python-physics/` system is reached by
`make python_tests` — `pytest.ini` is `testpaths = python/Tests` — so nothing in
CI runs them, and their READMEs carry the status instead.

### Type stubs

`python/manta/_manta.pyi` is **generated** by `make stubs`; `make stubs-check`
diffs it against a fresh generation and is what CI runs, because a stale stub is
worse than none — it reports the old signature as fact.
`python/manta/__init__.pyi` is hand-written and covers the Python layer, chiefly
the hook signatures a case implements.

Three things to know:

* Every hook declaration carries `# type: ignore[override]`, and that is load
  bearing rather than a workaround. The bound base method's signature is the C++
  one — the derivative hooks take a `VectorRef` out-parameter, and pybind11
  widens `int` to `SupportsInt | SupportsIndex` — while a Python case writes
  `(i, state, x, t)` and *returns* the vector. The declarations are what a
  user's subclass is checked against, which is the point; `warn_unused_ignores`
  is on, so the ignores cannot outlive the asymmetry.
* `check_untyped_defs` is what makes any of this bite. A physics case is
  unannotated Python and mypy skips unannotated defs without it.
* Two bindings were removed to keep the generated stub valid:
  `TransportSystem::ScalarGPrime` and `ComputePhysicsDerivatives` both take
  `GlobalStateMatrix`, which has no Python type, so the bound base methods were
  never callable from Python — they only put unresolvable names in the stub.
  `registerPhysicsCase` is `def`d after `TomlValue` for the same reason:
  pybind11 renders a `std::function` parameter from the types registered at that
  point, and binding it earlier left the raw toml11 C++ name in the signature.

### Out-of-tree builds

`make install PREFIX=...` installs the headers under `$PREFIX/include/manta`,
`libmanta.so`, and `manta.pc`. A physics case built as a shared object and named
in the config's `PhysicsPlugins` array is dlopened by `runManta` before
`InstantiateProblem`; its static initialiser registers it into the same
process-global map. Two traps, neither of which is a link error:

* **A plugin must be compiled with the flags `pkg-config --cflags manta`
  reports.** Eigen aligns its types to the widest vector unit the compiler knows
  about and inlines its expression templates into both sides of the boundary, so
  a plugin built without the core's `-march=` faults inside an aligned AVX-512
  load (`_mm512_load_pd`) the first time the solver touches its state. `manta.pc`
  records the *concrete* architecture (`-march=znver4`, not `native`) so a
  mismatch is a compile error rather than a run-time crash. `-DEIGEN_USE_BLAS`
  travels for the same reason.
* **A plugin must not link `-lmanta`.** The solver links the core objects
  directly, so a plugin that also pulled in `libmanta.so` would get a second copy
  of `PhysicsCases::map` and register into a map the solver never reads —
  silently. Compile against the headers alone and let the loader bind the MaNTA
  symbols to the host, which is linked `-rdynamic` for exactly that.
  `libmanta.so` is for embedding the solver in another program.

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

**With `spatialParameters = true` the parameter vector is indexed by node, and
`G_p` is `(ng * nCells * (k+1), np)` — one row per (objective, node), one column
per parameter *field*.** Two places in `computeAdjointGradients` fill that matrix
and they have to agree on the orientation: the explicit term assigns `dGFndp`
into a `(nPoints, np)` block (`SystemSolver.cpp:1675`), and the adjoint term
writes `G_p(gIndex * nPoints + node, pIndex)` (`SystemSolver.cpp:1814`). The
Python hook underneath, `dgFndp`, returns the *other* orientation, `(np,
nPoints)` — which is what the non-spatial branch of `PyAdjointProblem::dGFndp`
indexes as `dgdp(p, ind)` — so the spatial branch has to transpose, and for a
long time did not. Every spatial run aborted inside Eigen's assignment, and
would have transposed the gradient *silently* had `np` ever equalled the node
count. Note `checkShapeAndSet` is a plain assignment outside a `DEBUG` build, so
the only diagnostic was Eigen's own `resize()` assertion, naming
`Block<Matrix<double,-1,-1>,-1,-1,false>` and nothing about MaNTA.

Only Python can set the flag (`Python.cpp:307` exposes it on
`PyAdjointProblem`; the C++ base defaults false and `AdjointProblemTests.cpp:322`
asserts as much), so this is a trampoline concern rather than a solver one.
`Superconvergent = true` with spatial parameters throws rather than guessing —
the star node set would redefine how many parameters there are.

The spatial gradient is now exact per node as well as in total, both pinned in
`test_adjoint.py`. Getting there turned up the more general trap below.

**The adjoint's `dG/dZ` is `diag(w)`, not the mass matrix, and the difference is
invisible in every aggregate.** `GFn` reports `G = Σ_m w_m g(Z_m)` — the
interpolatory quadrature, `∫ I_h[g] dx` — so the exact derivative in the nodal
coefficient `Z_i` is `w_i dg/dZ|_i`. `DerivativeSubVector` used to apply
`InterpolateOntoBasis`, i.e. the cell mass matrix `M`. Since **`M·1 = w`
exactly** — a mass matrix's row sums *are* the quadrature weights — the two
agree whenever `dg/dZ` is constant across a cell and otherwise differ by
`(M − diag(w))·dg/dZ`, an operator that annihilates constants.

That last property is the whole reason it survived: the error summed to zero
over every cell, so `G_p.sum()`, the scalar-parameter gradient (exact to 7e-16
against a closed form) and every finite-difference check in the suite were
untouched. It was reachable *only* per node — i.e. only through spatial adjoint
parameters — where it appeared as an error set purely by the intra-cell node
index, symmetric and alternating in sign, decaying as O(h^4). That decay is a
convergence rate of an inconsistency, not a discretisation order: refining the
mesh hid it and nothing removed it. The generalisation worth keeping is that
**two quadratures agreeing is not evidence either is right** — the fixture that
should have caught this instead pinned the defect, because it differenced an
exactly-integrated `GFn` that no real case reports.

The pointwise `DerivativeSubVector` overload and the `dGdu_Vec`/`dGdq_Vec`/
`dGdsigma_Vec` wrappers over it are gone — they computed `∫ dg/dZ φ_i dx`, the
derivative of `∫ g dx`, and no solve ever called them. `dGdaux_Vec` was the last
one left and is now the same operator over `nAux` blocks: it takes the nodal
`dg/dphi` from the batched `dg` and weights it, and `dGdt` goes through it too
rather than applying the mass matrix inline. A C++ case's `dgFn_dphi` still
reaches it, through `AdjointProblem::dg`'s default, which samples the hook at the
nodes; a Python case supplies `dg` and `PyAdjointProblem::dgFn_dphi` raises.

**Beware how nearly `diag(w)` and `M` agree — it is far more than the constants
the argument above needs.** `(M v)_i = ∫ φ_i v` and `(diag(w) v)_i = v_i ∫ φ_i`,
so they coincide whenever the interpolatory rule integrates `φ_i v` exactly; on
`k+1` Chebyshev points of the first kind that rule is symmetric, hence exact to
degree `k+1` for even `k`. At `k = 2` — which is what the adjoint fixtures use —
that covers every *affine* `dg/dZ`, and the mocks' hooks are affine in `x`. Both
`the_derivative_sub_vector_weights_dg_by_the_integration_weights` and its aux
sibling therefore passed with the mass matrix reinstated, by 3e-16 and 5e-16,
until each was given a second half driven by a synthetic degree-`k` `dg/dZ` and a
guard that the two operators still differ on it. Before that the only case in the
suite that noticed at all was `dGdt_matches_a_finite_difference_of_the_objective`,
at a relative 6e-6 against a 1e-6 tolerance. A reference built "straight from the
weights" pins the formula, not the operator, if the data cannot tell them apart.

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
* **`extern/autodiff` points at a fork, and `git submodule update --remote` would
  silently undo that.** Upstream `autodiff/autodiff` specialises `VectorTraits` on
  `Eigen::internal::SingleRange` as a plain type, which Eigen 5.0 made a template,
  so upstream does not compile against Eigen 5 *at all* — and its `main` has not
  moved since January 2025, i.e. predates Eigen 5. The submodule therefore tracks
  `ianabel/autodiff` branch `eigen-5-singlerange`, which carries that one patch;
  `.gitmodules` records why. Upstream PR autodiff/autodiff#397 is the thing to
  watch: when it merges, point the submodule back and delete the fork. Until then
  a plain `--remote` update reverts to a commit that cannot build with Eigen 5,
  and the failure appears as a wall of template errors inside a third-party
  header rather than as anything about submodules.

* **Eigen 3.4.x and 5.0.x are both supported, and `EIGEN_VERSION_AT_LEAST` cannot
  tell them apart.** Eigen 5.0 moved to semver by keeping `EIGEN_WORLD_VERSION` at
  3 forever and renumbering the rest, so `EIGEN_MAJOR_VERSION` went 4 -> 5 and the
  macro's arguments changed meaning underneath it: it compares
  `(WORLD, MAJOR, MINOR)` in 3.4 and `(MAJOR, MINOR, PATCH)` in 5.0. So
  `EIGEN_VERSION_AT_LEAST(3, 3, 90)` is *true* under Eigen 5 — it reduces to
  `5 > 3` — which is how `extern/autodiff` came to compile a block guarded against
  old Eigens into a version that cannot accept it. **Use `EIGEN_MAJOR_VERSION >= 5`
  to branch on the major version**, never `EIGEN_VERSION_AT_LEAST`.

  Two things moved that MaNTA cared about. `Eigen::all` is now only
  `Eigen::placeholders::all`, and in 3.4 that spelling exists but is
  `EIGEN_DEPRECATED` — so with `-Werror` neither spelling compiles on both, and the
  warning fires at *our* call site, where `-isystem` cannot suppress it. Every use
  was a `.row()`, `.middleCols()` or `.leftCols()` written the long way and is now
  spelled that way, so no version branch is needed in this tree. And
  `internal::SingleRange` became a template, which is the autodiff patch above.
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
* **gcc and clang do not diagnose the same things, so build with clang
  occasionally** — that is what CI's clang matrix legs are for. (CI is seven
  `build-and-test` legs: g++-14/15/16 and clang++-19/20/21 against the distro's
  Eigen 3.4.0, plus g++-14 against Eigen 5.0.1; then a Fedora container job that
  only *compiles*, to keep the build's notions of a system prefix — `/usr/lib64`,
  pkg-config-discovered netCDF — from quietly becoming Ubuntu-specific.) gcc never
  diagnoses a polymorphic base with a non-virtual destructor; clang does
  (`-Wdelete-non-abstract-non-virtual-dtor`), and it reports it at the point of
  *destruction* inside libstdc++, once per instantiating translation unit, which
  makes the message look like a standard-library problem rather than yours. That is
  how `MagneticField`'s missing virtual destructor was found.

  Until `283b9a3` this entry also warned that `-Wno-parentheses` was applied
  globally and that on gcc it silently takes `-Wdangling-else` with it, so gcc
  could not report a dangling `else` and only clang would. **That suppression is
  gone**, along with `-Wno-deprecated-literal-operator`; `-Wall` enables
  `-Wparentheses`, so gcc now diagnoses a dangling `else` like everything else,
  and under `-Werror` it is an error rather than a warning.

  Worth keeping the reason they went, because it generalises. Both existed to
  silence *third-party* headers — `-Wno-deprecated-literal-operator` for toml11's
  `operator""_toml`, declared without the space C++23 wants — and both became
  unnecessary once those headers moved to `-isystem`, which suppresses their
  warnings at source. What they went on doing in the meantime was hiding defects
  in *our* code: a global `-Wno-` outlives whatever it was added for, and nothing
  reports that it has stopped earning its place. Prefer `-isystem` on the
  dependency to a blanket suppression on the project.
* **With clang, the libstdc++ version is part of the configuration.** clang selects
  the newest GCC installation on the box, so a local clang build and CI's clang
  legs need not use the same standard library: CI gets the `ubuntu-24.04` image's
  libstdc++ 14 (14.2.0), a box with g++-15 installed gets libstdc++ 15. So "it
  builds with clang here" is weaker evidence than it looks. Pin the library to
  check portability: `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/14`, or
  extract noble's `libstdc++-14-dev` .deb and point `-nostdinc++ -isystem` at it
  when the patch level matters.

  It has cost one red CI already. `std::bind_front` converted to `std::function`
  is rejected by clang ≥ 20 with libstdc++ < 14.4: libstdc++ gives
  `_Bind_front::operator()` two implementations chosen by
  `#if __cpp_explicit_this_parameter`, clang defines that macro only from clang 20,
  and the explicit-object-parameter branch failed `std::function`'s `_Callable`
  probe until 14.4. `SystemSolver::setInitialConditions` and `DGSoln::AssignU` use
  lambdas rather than the bind family for that reason — don't reintroduce it.
* **Third-party includes use `-isystem`, not `-I`** (`Makefile.config`: SUNDIALS,
  toml11, Boost, netCDF, Eigen, autodiff, and pybind11 in the top-level Makefile).
  `-Werror` is on, and Eigen's own headers do trip `-Wunused-but-set-variable`
  under clang — reachable only from the pybind11 build, which pulls in
  `SparseCore`. Adding a dependency with `-I` re-arms that.
* **...but never `-isystem` a directory the compiler already searches.** Go through
  `$(call sysinclude,DIR)` in `Makefile.config`, never a bare `-isystem`. Passing a
  default system directory is not a no-op: gcc and clang both de-duplicate it,
  dropping the directory from its proper place at the *end* of the system chain and
  searching it where the `-isystem` appeared — ahead of the libstdc++ headers.
  `<cstdlib>` then does `#include_next <stdlib.h>`, which only considers directories
  *after* the one holding it, so every translation unit dies with

  ```
  /usr/include/c++/16/cstdlib:83:15: fatal error: stdlib.h: No such file or directory
     83 | #include_next <stdlib.h>
  ```

  `NETCDF_DIR=/usr` did exactly that, which is what a package-manager install means
  on Debian/Ubuntu, and `EIGEN_DIR=/usr/include` does the same. Note the asymmetry:
  `-I` is safe here — gcc documents that an `-I` naming a standard system directory
  "is ignored. The directory is still searched but as a system directory at its
  normal position" — so the fix is to filter, not to downgrade to `-I`, which would
  re-arm the `-Werror` problem above. **`NETCDF_DIR`/`NETCDF_CXX_DIR` should be
  unset for a system install**; with neither set, `Makefile.config` asks pkg-config.
  `sysinclude` compares canonically because clang reports its C++ directories as
  `/usr/lib/gcc/x86_64-linux-gnu/16/../../../../include/c++/16` where gcc reports
  `/usr/include/c++/16`, and a probe that fails filters nothing — degrading to the
  old behaviour rather than to a new error. It cannot be replaced by a "does this
  directory hold the header we want" test: `/usr/include` really does hold
  `netcdf.h`. CI's `Makefile.local` leaves `NETCDF_DIR` unset, which is the one
  configuration where this is invisible, so a workflow step compiles one object with
  `NETCDF_DIR=/usr` on every matrix leg — on every leg because the probe is a
  compiler command whose output format differs between gcc and clang.
* **A comma inside `$(if ...)` is an argument separator, not text.** `syslibdir` in
  `Makefile.config` writes `-Wl$(comma)-rpath` rather than a literal `-Wl,-rpath`
  for that reason. Spelled literally, make reads the body of
  `$(if $(strip $(1)),-L$(1) -Wl,-rpath $(1))` as *then:* `-L$(1) -Wl` and *else:*
  `-rpath $(1)`, so an empty argument emitted a bare `-rpath` that swallowed the
  next flag and a real one silently lost its rpath. The `$(comma)` looks like
  clutter and is load-bearing; don't inline it.
* **`make -B` does not work in this tree.** `--always-make` tries to remake every
  target including the included `Makefile.local`, whose rule is a bare
  `$(error You need to provide a Makefile.local...)`, so `-B` fails immediately with
  that message no matter what you asked for. To see the recipe for an
  already-built target, delete it, or read the expanded variables from a throwaway
  makefile that `include`s `Makefile.config` and `$(info)`s them.
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
* **`RF_cellwise` and `L_global` hold *time-dependent* boundary data, and only
  `updateBoundaryConditions(t)` may fill them.** `initialiseMatrices` sizes and
  zeroes them; `setInitialConditions` calls `updateBoundaryConditions(t0)` before
  solving the initial `dydt` out of them, and `residual` / `updateMatricesForJacSolve`
  refresh them at their own time. `initialiseMatrices` used to fill them itself at
  a hardcoded `t = 0.0`, which was wrong twice over: a run with `t0 != 0` built its
  initial condition from the wrong boundaries, and because `initialize()` skips
  `initialiseMatrices` when already initialised, a second run on the same solver
  solved its initial `dydt` out of the *previous run's final-time* boundary values.
  That second effect is half of what made a second integration fail — see
  `Tests/README.md`. Nothing in the tree notices a `t0` error, because every
  fixture starts at zero, so `the_initial_condition_uses_boundary_data_at_t0` is
  the only thing standing between that and a silent return.
* **`dydtComplete` is deliberately not IDA's `dYdt`, and the duplication is the
  point.** `AlgebraicDerivatives.cpp` solves the differentiated algebraic
  constraints for `q'`, `sigma'`, `phi'` and `lambda'` — IDA never computes them,
  because `IDA_YA_YDP_INIT` produces algebraic *values* and differential
  *derivatives* — and writes the answer into its own vector. Folding it back into
  `dYdt`, which is the obvious tidy-up, would change the state IDA takes its first
  step from: the surviving symptom would be a step-size or convergence failure
  somewhere later in the run, pointing nowhere near here. Only
  `objectiveIsDecreasing()` reads it, and only a run that arms the gate pays for
  it, so nothing else notices either way.
* **The differential rows of that solve are the identity on purpose**, and so are
  the Dirichlet trace rows. `u'` and a differential scalar's `mu'` are *data* —
  IDA has them — so their rows carry `1` and the known derivative rather than a
  differentiated equation, which would bring in `u''`. The Dirichlet trace rows
  look like a redundant special case and are not: `residual` never writes them
  (`lambda = g_D(t)` is imposed inside the linear solve, which is also why a
  finite-differenced Jacobian is rank-deficient by exactly the number of Dirichlet
  boundaries), so without their own identity row and `dg_D/dt` the matrix is
  singular by that same count. `the_u_block_round_trips_through_the_identity_row`
  covers the first; the second is what
  `the_derivatives_match_a_manufactured_solution` checks through `lambda'`.
* **The central-difference step there is `cbrt(eps)`, not `sqrt(eps)`.** `sqrt(eps)`
  is the *one-sided* choice, where truncation is `O(h F'')`; a central difference
  has truncation `O(h^2 F''')` against round-off `O(eps |F| / h)`, and those
  balance at `eps^(1/3)`. Using `sqrt(eps)` leaves round-off at `eps/h = 1.5e-8`
  against a truncation of `2e-16` — eight orders apart rather than comparable — and
  it measurably costs 2.5 decimal places: the manufactured case gets `q'` to 3.4e-8
  with `sqrt(eps)` and to 5.6e-11 with `cbrt(eps)`, on a problem whose explicit
  time dependence is linear in `t` and therefore has *no* truncation error at any
  step. The design document specified `sqrt(eps)` and called it the central
  choice; it isn't.
* **`OutputFilename` names the output, and only its *basename* survives.**
  `loadSolverConfig` fills it from the config file's stem when the key is absent,
  so a run still defaults to `myrun.conf` -> `myrun.nc`; `Solver.cpp` then takes
  `inputFilePath.filename()` of it. `filename()`, **not** `stem()`: the value it
  is given is already a base name, and stemming it a second time would turn
  `run.v2.conf` into `run.nc`. The directory part is dropped, so `.nc` / `.dat` /
  `.restart.nc` still land in the current directory whatever path
  `OutputFilename` carries — `OutputFilename = "runs/case7"` writes `./case7.nc`.
  That is pinned by `test_output_filename_keeps_only_the_basename`, which records
  it as behaviour to change deliberately rather than by accident. An explicit
  `RestartFile` is *not* filtered that way and is opened as given.
* **Not every `.nc` in the tree is output.** `clean_data` sweeps generated data
  from the directories in `CLEAN_DATA_DIRS`, sparing `*.ref.nc` / `*.ref.dat`.
  `Tests/UnitTests` is deliberately absent: its data files are tracked test
  *inputs* — `testic.nc` (`AutodiffTest.cpp`) and `MatrixDiffusion.restart.nc`
  (`SystemSolverTests.cpp:378`) — and the second has no `.ref.` in its name, so
  the keep-pattern would not save it. Check tracked status, not the filename,
  before adding a directory there. Unit-test output itself lands at the repo
  root, because `make test` runs the binary from there. `python` is absent for a
  different reason: since the drivers moved out to `python-examples/` and
  `python-physics/`, nothing writes output there. `.h5` and `.pkl` are not in the
  pattern list either — the DESC equilibria under `python-physics/stellarator/`
  are expensive, and `python-physics/mirror-plasma/land.pkl` is generated once by
  `landremann.py` rather than by a run.
* **Unanchored `.gitignore` patterns match at every depth.** The root scratch
  entries (`Plots/`, `runs-for-bhavin/`, `scalar-tests/`, `toy-model/`) are
  written with a leading slash for that reason: unanchored, `toy-model/` also
  ignored `python-examples/toy-model/`, so the example was left out of a commit
  with nothing to say so. `git check-ignore -v <path>` names the line
  responsible.
* **`printSources` reads the source cache through a basis of the residual's
  order.** With `Superconvergent = true` the cache holds `k+2` values per cell
  rather than `k+1`, so `SystemSolver::print` picks its basis and stride from the
  flag; hardcoding `k+1` there reads across cell boundaries.
* **netCDF is the default output; the `.dat` files are opt-in.** A run writes
  `<stem>.nc` and `<stem>.restart.nc` unless `WriteOutput = false`, which gates
  every netCDF and restart write in `Solver.cpp`. The plain-text gnuplot output
  needs `WriteDatFile = true`, and `<stem>.dydt.dat` / `<stem>.res.dat` need
  `WriteDebugDatFiles = true` *and* a `PHYSICS_DEBUG` build. Those two are
  deliberately **not** nested under `WriteOutput`: they are opt-in already, so a
  config asking only for `WriteDatFile` gets it. All three are accepted by
  `runManta` and `PyRunner::configure` alike — `WriteOutput` was read into an
  unused local on the Python side and read by nothing at all on the TOML side
  until the schema landed, while nine test call sites passed `WriteOutput: False`
  and went on writing the files they believed they had suppressed.
* **Tests reach private `SystemSolver` members** through `MANTA_TEST_PRIVATE`,
  which a `-DTEST` build widens to `public`. No friend declarations needed.
* **The extension's ABI suffix comes from `PYTHON_CONFIG`, and the venv need not
  agree with it.** `make python` names the module from
  `$(PYTHON_CONFIG) --extension-suffix` and takes its headers from the same
  program, so the two always match each other — but not necessarily the interpreter
  that will import them. `PYTHON_CONFIG` prefers a `pythonX.Y-config` matching
  `.venv`, and falls back to plain `python3-config` when there is none; that follows
  the distribution's unversioned `python3` symlink. On a box whose `python3` has
  moved ahead of the venv, the fallback builds `_manta.cpython-314-*.so` while
  `.venv` runs 3.13, and `make python` *succeeds* while `python_tests`,
  `stubs-check` and `typecheck` all fail — each with a message pointing somewhere
  else. pytest exits "manta package not importable. Build it with `make python`",
  which you just did. `typecheck` reports an `ImportError` for `_manta` dressed up
  as "most likely due to a circular import", which sends you into `__init__.py`.
  `stubs-check` is the worst of the three: regenerating the stub needs the import
  too, so it fails to write one and then reports `_manta.pyi is stale -- run 'make
  stubs' and commit the result`, which is a claim about a committed file that is
  in fact fine. Check `ls python/manta/*.so` against
  `python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'`
  before believing any of those three failures. The fix is the matching
  `pythonX.Y-dev` package, or `make venv VENV_PYTHON=... VENV_CREATE_FLAGS=--clear`;
  the header directory alone is not enough, since a `/usr/include/python3.13` left
  behind by other packages can exist with no `Python.h` in it. `Makefile:34-51`
  documents the mechanism, and `make python PYTHON_CONFIG=pythonX.Y-config`
  overrides it — but only if that program is installed, because `pythonX.Y-config`
  derives its prefix from `argv[0]`.
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
* Restarting is fragile at tight tolerances, more so with `nAux > 0`; each
  regression round-trip case runs at the tightest tolerance that completes.
* ~~`python/Tests/test_reference_solutions.py::test_jax_aux_test` is a
  `strict=True` xfail.~~ Fixed. It was four faults in `manta.jax`, all reachable
  only with `nAux > 0`: `AuxGPrime` and `dAux_dp` take an extra argument ahead
  of the state and so need `ShiftedState_Decorator` rather than
  `MaNTA_Decorator`; `dgFn_dphi` was undecorated and indexed its result as a
  dict; and `JAXAdjointProblem` never bound `transport_system.aux`. See
  `Tests/README.md`, and note that the symptom the xfail *recorded* was not the
  symptom it *had* by the time it was fixed — re-run before theorising.
* **There is no C++ mirror plasma any more.** `PhysicsCases/MirrorPlasma.{cpp,hpp}`,
  `PhysicsCases/MirrorPlasma/` (`AmbipolarPhi`, `ConstantVoltage`,
  `PlasmaConstants`, `PlasmaDiagnostics`) and `PhysicsCases/CurvedMirrorPlasma/`
  were removed: `python-physics/mirror-plasma` is the implementation that is
  developed now, and two of them was one too many. `MirrorPlasmaTest.cpp`, the
  nine `Config/*.conf` files that selected it, `CylindricalMagneticField.py` (it
  generated the `Bfield.nc` only `useNcBField` read) and `util/mirror_plots.py`
  (it read `Var0`-style groups, so it had already been broken by the
  name-your-own-variables change) went with it.

  Three things survive that were shared. `PhysicsCases/MagneticFields.{cpp,hpp}`
  stays — `AdjointPlasma` includes it, though its field member is commented out,
  and `MagneticFieldTest.cpp` is now the *whole* of its coverage rather than a
  supplement, since nothing instantiates either field class. `AdjointPlasma`
  itself is untouched: every `PlasmaConstants` use in it was already commented
  out. And `AutodiffTransportSystem::MMS_Solution` remains as a facility with no
  overrider left — `MirrorPlasma` was the only one.

  The cost is real and worth naming: `plasma_init_tests` and
  `neutral_model_tests` are gone from CI, and the Python case's own suite is not
  run by `make python_tests` (`pytest.ini` is `testpaths = python/Tests`) because
  it needs `desc` and `optimistix`. `Tests/README.md` records this.
* The `UseMMS` options on `LinearDiffusion` and `LinearDiffSourceTest` have been
  removed: the first's manufactured solution did not satisfy its own boundary
  conditions, and the second never applied `MMS_Source` at all. Order of accuracy
  is measured by `Tests/UnitTests/MMSConvergenceTests.cpp`, which builds its own
  manufactured problems.
