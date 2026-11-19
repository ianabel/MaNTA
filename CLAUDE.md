# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MaNTA (Maryland Nonlinear Transport Analyzer) solves 1-D reaction–diffusion /
transport systems with a hybridizable discontinuous Galerkin (HDG) spatial
discretisation, integrated in time as an index-1 DAE by SUNDIALS IDA.

This file is the summary: what is needed on almost every task. The detail, the
measurements behind each decision and the traps in each subsystem are in the
topic files below; **read the one for the area you are about to change before
changing it.**

| File | Read it before touching |
|---|---|
| `docs/dev/build.md` | compiler flags, CMake, dependencies, out-of-tree plugin builds; any compiler-, BLAS- or link-specific failure |
| `docs/dev/ci.md` | a CI leg (added, dropped, renamed) or the branch-protection rule |
| `docs/dev/solver.md` | `Solver.cpp`, `SteadyState.cpp`, `Bounds.cpp`, initial conditions, restarts, sources reading `du/dt`; reading an IDA failure |
| `docs/dev/parallelism.md` | `MANTA_OPENMP`, `util/ParallelFor.hpp`, BLAS threading, the trace solve (`util/BandedMatrix.hpp`) |
| `docs/dev/configuration.md` | `ConfigSchema.cpp`, `SolverConfig`, a new or renamed key, the output files |
| `docs/dev/discretisation.md` | `Superconvergent`, `Postprocessing`, `tauScaling` |
| `docs/dev/physics-cases.md` | `SystemSpec`, boundary conditions, case registration, the evaluation plan, regrids |
| `docs/dev/field-coupling.md` | `FieldModel`, `EquilibriumModel`, the coupling tiers, the field solves |
| `docs/dev/python.md` | `Python.cpp`, the trampolines, `PyRunner`/`SteadySolve`, `manta.jax`, the stubs, the staged package |
| `docs/dev/adjoints.md` | anything adjoint: gradients, spatial parameters, the discretisation estimate |
| `Tests/README.md` | adding tests, interpreting a coverage number; the known gaps |
| `TODO`, `PERFORMANCE.md`, `MESH-REFINEMENT.md`, `FEATURES.md` | open work, timings, mesh adaptation, the feature list |

`README.md` covers first-time setup (dependencies, `CMakeUserPresets.json`,
installing SUNDIALS); `docs/*.rst` is the user documentation.

## Commands

**The build is CMake, out of source.** Machine-specific paths go on the configure
line or into a gitignored `CMakeUserPresets.json`.

```sh
cmake --preset default    # configure into build/ (Release; also: debug, coverage, portable)
cmake --build build -j    # solver + libmanta.so + the Python module + the unit tests
ctest --test-dir build    # all three suites

cmake --build build --target MaNTA        # the solver only, at build/MaNTA
cmake --build build --target _manta       # the manta package, at build/python/manta
cmake --build build --target UnitTests    # build/Tests/UnitTests/UnitTests
cmake --install build --prefix ...        # bin/, lib/ (+ pkgconfig, the Python package),
                                          # include/manta, share/doc; DESTDIR honoured
export PYTHONPATH=$PWD/build/python       # use the build's package in place
pip install .             # the `manta` package and the `manta` console script
pip install .[jax]        # ...and manta.jax (jax, equinox, jaxtyping)

cmake --build build --target unit_tests | regression_tests | python_tests
cmake --build build --target docs         # build/docs/html, via build/venv-docs built from
                                          # docs/requirements.txt; -W, as Read the Docs runs it
cmake --install build --component docs    # ...and copy it to docs/_build/html
cmake --build build --target stubs        # generate _manta.pyi into build/stubs/
cmake --build build --target stubs-check  # fail if the committed stub differs (CI runs this)
cmake --build build --target stubs-update # copy the generated stub over python/manta/_manta.pyi
cmake --build build --target typecheck    # mypy over the manta package
cmake --build build --target venv         # build/venv from requirements.txt, plus gcovr
cmake --build build --target clean_data   # run output (.nc/.restart.nc/.dat) at the root and in
                                          # Tests/RegressionTests, python/Tests/ and each
                                          # directory under python-examples/ and python-physics/
cmake --build build --target clean_coverage

cmake --preset coverage                            # a build directory of its own
cmake --build build-coverage --target coverage     # all three suites instrumented + gcovr

build/MaNTA --list-options   # every configuration key, straight from ConfigSchema.cpp
```

There is no `clean` target and no need for one: `rm -rf build`. **A build writes
nothing outside its build directory, and the suites run from there too** — CI
checks it, with `git status --ignored` after the suites. `stubs-update` and the
docs component of `install` are the only things that write into the checkout,
and both exist to. Keep it that way: anything new that a target or a test
writes goes under `PROJECT_BINARY_DIR`.

The regression and Python suites need `requirements.txt` installed, but **nothing
needs to be on `PATH`**. CMake finds one interpreter — preferring `build/venv`,
then a `.venv` in the repo root — and runs the regression driver and pytest with
that one. Name a different one once, with `-DPython3_EXECUTABLE=...`.

Running one test:

```sh
build/Tests/UnitTests/UnitTests --run_test=solve_jac_tests/solve_hdg_jac_agrees_with_a_dense_solve --log_level=all
build/Tests/UnitTests/UnitTests --run_test=mms_convergence_tests --log_level=message   # see BOOST_TEST_MESSAGE output
build/python/Tests/run-pytest python/Tests/test_adjoint.py::test_adjoint_gradient_matches_finite_differences
build/Tests/RegressionTests/run-regression --tolerance 1e-2
ctest --test-dir build -R '^unit$' --output-on-failure
```

`run-pytest` and `run-regression` are generated into the build tree and are what
CTest runs, so a test run by hand from any cwd runs exactly as it does there:
from the build tree, against that build's package, with pytest's cache, its
temporary directories and the tests' bytecode kept out of the source tree. A bare
`pytest` from the repo root does none of that. New unit-test `.cpp`
files must be added to `MANTA_TEST_SOURCES` in `Tests/UnitTests/CMakeLists.txt` —
kept an explicit list, unlike `PhysicsCases/*.cpp`, which is globbed with
`CONFIGURE_DEPENDS`.

Build variants are build types and `MANTA_*` options, not command-line variables:
`--preset debug` (`-O0 -g -DDEBUG -DPHYSICS_DEBUG`, and `State.hpp`'s
`checkShapeAndSet` becomes shape-checking rather than a plain assignment),
`--preset coverage`, `MANTA_OPENMP` (threads the cell-independent loops through
`util/ParallelFor.hpp` — read `docs/dev/parallelism.md` before turning this
on), `MANTA_VERBOSE`, `MANTA_PHYSICS_DEBUG`,
`MANTA_NATIVE_ARCH`, `MANTA_ASSERTS`, `MANTA_TESTS`, `MANTA_PYTHON`, `MANTA_XLA_FFI`/`MANTA_CUDA`
(JAX FFI, needs jaxlib headers). `cmake -B build -LH` lists them all.

**A Release build keeps `assert()` and Eigen's assertions** (`MANTA_ASSERTS`,
default on, strips the `NDEBUG` CMake would add). They cost nothing at k = 3-4
and 7-9% at k = 8-10, so **a timing claim that does not state `k` is
under-specified**. `docs/dev/build.md` has the table and the build files.

## Branch protection on `main`

**`main` is protected: work goes on a branch and merges through a pull request.**
No approving review is required, so a PR merges as soon as its required checks
are green; `strict: true`, so it must be up to date with `main` first. Thirteen
of `ci.yml`'s fifteen job names are required contexts (not `Coverage`, not the
AlmaLinux leg). Read the live rule rather than trusting this paragraph:
`gh api repos/ianabel/MaNTA/branches/main/protection`.

**A required context is matched by the job's rendered name**, so adding,
dropping or relabelling a leg changes what the rule must list, and a context
nothing publishes blocks every PR silently. Update the rule in the same change,
only once the leg is on `main`, and check the two agree; `docs/dev/ci.md` has
the `diff` that does it.

## Working on this repository

Traps in the surrounding tooling rather than in the code. Every one of these has
cost real time here, and none of them announces itself.

* **`gh` on this box is too old for `gh pr checks --json`**, and it fails by
  printing usage to *stderr* and nothing to stdout. A poll loop built around it
  therefore never sees a terminal state and waits forever, in silence — the shape
  of bug where the absence of a notification is indistinguishable from "still
  running". `gh pr view <N> --json statusCheckRollup` works and is the one to
  use. `gh pr create` and `gh pr edit` fail differently again, on a
  Projects-classic GraphQL deprecation; open and edit PRs through
  `gh api repos/ianabel/MaNTA/pulls` instead. **Run any `gh` subcommand once and
  look at what it actually returns before building a wait loop on it.**

* **`git diff main <branch>` does not answer "does this branch hold work I would
  lose".** For a branch that is fully merged but sitting at an older point, that
  diff reports `main`'s *subsequent* commits backwards: a merged branch here
  showed "89 files changed, 7917 insertions(+)", which reads like unmerged work
  and is the opposite. The question is `git rev-list --count main..<branch>`,
  which is 0 for exactly the branches that are safe to delete. `git branch -d`
  applies that same test and refuses when it fails, so prefer it to `-D` and let
  git hold the veto.

* **`git reset` clears `MERGE_HEAD`.** Unstaging something in the middle of a
  conflicted merge turns the merge into an ordinary one-parent commit, so the
  merge silently does not happen — and GitHub then reports the PR `CONFLICTING`
  against a `main` whose content it appears to contain already. Rebuilding the
  commit with `git commit-tree $TREE -p <ours> -p <theirs>` fixes it without
  redoing the resolution; confirm with an empty `git diff` against the broken
  commit and `git merge-base --is-ancestor origin/main HEAD`.

* **`git add -A` is unsafe on a branch based on `main`.** `.gitignore` covers
  `build/` but not `build-*/`, and this tree accumulates `build-omp/`,
  `build-debug/` and `build-coverage/`. Stage by name.

* **A conflict resolved by splicing strings damages prose in ways nothing
  compiles.** Two reached `main` that way: a `Solver.cpp` comment joined
  mid-sentence into "does not apply about it does not either", and a blank line
  dropped before a reST label, which deletes the label outright. The compiler
  sees neither. The docs one is caught by the `docs` target, which builds with
  `-W`, and by CI's `Docs` leg, which runs the same script
  (`cmake/MantaDocs.cmake`); a comment is caught by nothing. Read every resolved
  hunk that is prose, and run `cmake --build build --target docs` before pushing
  anything touching `docs/` rather than waiting for the leg.

* **A clean rebase can still revert a fix.** When the branch and the commits it
  replays onto both touch a file, "no conflicts" means git found no textual
  overlap — not that the result is what you want. `git range-diff <old-base>..<old>
  <new-base>..<new>` reports whether each patch survived unchanged, `=` on every
  line being the thing to look for, and then read the merged region itself. This
  matters most for a file a *bugfix* branch also touched, which is precisely when
  the rebase is least likely to conflict and most likely to matter.

## Architecture

### The equation being solved

A physics case defines, per variable `i`:

```
a_i d_t u_i + d_x sigma_i = S_i(u, q, sigma, phi, du/dt, x, t)
sigma_i    = sigma_hat_i(u, q, x, t)          # the flux
q_i        = d_x u_i                          # introduced as an unknown
G_j(phi, u, q, sigma, x) = 0                  # nAux algebraic auxiliary constraints
G_s(mu, y, dy/dt, t)     = 0                  # nScalars global (non-spatial) unknowns
R_m(psi, dpsi/dt, y, t)  = 0                  # nField magnetic-field unknowns
```

**A source may read `du/dt`, and only `du/dt`** (`State::udot(j)`), and it is
zero unless some variable's `FieldSpec` sets `sourceReadsTimeDerivatives`. It
enters the Jacobian as `- alpha * dS/d(udot)` beside the mass term, so it is
invisible at `alpha = 0`. `docs/dev/solver.md` has the four consequences.

`sigma`, `q`, `u` and the auxiliary variables `phi` live per cell; `lambda` is
the HDG trace unknown on cell faces; `mu` are the global scalars; `psi` are a
field model's unknowns and are absent unless one is attached. That ordering —
**`[sigma | q | u | aux]` per cell, then all of `lambda`, then `mu`, then
`psi`** — is the DOF layout of both the solution vector (`DGSoln::Map`) and the
local Jacobian block `MX`, and getting a column index wrong there is the most
common way to break the solver silently. `DGSoln::getDoF()` is the one authority
on the total length: the formula was open-coded in three places, and the copy
that did not know about `nField` wrote a *short* restart file whose recorded
`nDOF` matched the uncoupled formula — so the truncated file read back as
consistent. Note that only `PhysicsCases/` may be physics; the core is generic.

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
`State::sigmaHat(i)` the physical flux `SigmaFn` returned. The autodiff layer
hands `Source` and `GFunc` the physical flux (`physicalFlux`) and negates
`dSources_dsigma` and `AuxGPrime`'s sigma block to match, since autodiff then
differentiates in `sigma_hat` while the Jacobian's column is the stored sigma.
No case in the tree reads it; `AutodiffSigmaTests.cpp` is what pins both halves
-- a manufactured solve for the value, which the wrong sign misses by O(1), and
a finite-differenced Jacobian for the derivative, which a missing negation
fails while the solve still passes.

### Solve path, in brief

`Solver.cpp:runSolver` drives three phases that can also be called separately:
`SystemSolver::initialize`, `integrate(tFinal)` and `destroySundials`.
`docs/dev/solver.md` has all of it; what matters on almost every change:

* **Steady solves have three tests.** `SteadyStateTest = "NewtonStep"` (the
  default) is MaNTA's own loop in `SteadyState.cpp`, on IDA's corrector test;
  `"Residual"` is the KINSOL path; `"Objective"` is the NewtonStep loop stopping
  on the named objectives' algebraic error. A steady solve can be taken in
  slices (`MaxContinuationSteps`, `continueSteadyState()`), and a sliced solve
  resumes bit for bit because everything carried between steps is a member.
* **The Jacobian is never assembled.** IDA gets a custom `SUNLinearSolver`
  (`SunLinSolWrapper`) and a deliberately empty `SUNMatrix`. `residual` does not
  write the Dirichlet boundary rows, so a finite-differenced Jacobian is
  rank-deficient by the number of Dirichlet boundaries.
  `updateMatricesForJacSolve` builds and factorises the per-cell `MX` blocks
  (`Matrices.cpp`); `solveHDGJac` condenses onto `lambda` and back-substitutes;
  `solveJacEq` borders that for the global scalars with a Schur complement,
  `(N + W^T A^-1 V) mu = s - W^T A^-1 r`, and **must never invert `N` on its
  own**: an integral controller's row makes `N` singular at an undamped step
  (`the_bordered_solve_does_not_invert_n_on_its_own`).
* **So an error in the Jacobian is slow Newton convergence, never a wrong
  answer.** The tests that see it are `SolveJacTests.cpp` (finite-difference the
  residual, require `J dy = g`) and `MMSConvergenceTests.cpp` (observed order).
* **The adjoint operator is the forward Jacobian's transpose, block for block**
  (`assembleAdjointOperator` against `assembleCellMatrix`). A block missing from
  the forward Jacobian only slows Newton; the same block missing from the adjoint
  is a silently wrong gradient beside a correct `G`. Add a coupling to both or to
  neither. `docs/dev/adjoints.md`.
* **A reused solver gives the same answer as a fresh one, bit for bit**
  (`a_second_integration_on_one_solver_matches_a_fresh_one`, zero tolerance --
  do not relax it). `initialize` skips `initialiseMatrices` on a reused solver,
  so anything computed there must be run-independent, and anything a run changes
  is reset in `initialize()`'s unconditional part. See the `RF_cellwise` trap
  below; the field models, the equilibrium cache and the tau state all follow it.
* **A physics call at a point set the evaluation plan does not list is a bug.**
  Adding a physics call means adding its site to `evaluationPlan()`
  (`docs/dev/physics-cases.md`). Only `PhysicsCases/` may be physics; the core is
  generic.
* **When IDA gives up, get its own account first**: SUNDIALS is built with
  logging, so
  `SUNLOGGER_INFO_FILENAME=/tmp/ida-info.log SUNLOGGER_DEBUG_FILENAME=/tmp/ida-debug.log build/MaNTA foo.conf`
  records every step attempt. `docs/dev/solver.md` decodes the return codes.
* **Inert-when-unused features are checked by `cmp`, not by the regression
  suite**, whose 5e-3 tolerance cannot see a change of that kind: build `main`,
  run both binaries over every `Tests/RegressionTests/*.conf` and compare the
  `.nc` and `.restart.nc` byte for byte. Bounds, field coupling, OpenMP and the
  `Constant` tau path all rest on it.

### Non-owning state views

`DGSoln` / `DGApprox` are **`Eigen::Map` views over memory SUNDIALS owns**, not
containers. `SystemSolver::y` maps the `N_Vector` that `initialize` allocates and
`destroySundials` frees, so it dangles after a run; `yJac`/`dydtJac` own their
memory (`yJacMem`) and are what outlives the solve. Anything reading "the
solution" after a run must use `yJac`. `initialize` seeds it with the initial
condition, so it is also valid *before* `integrate`, for a caller that stops
between the two.

`DGSolnImpl` holds its basis **by value** (`const BasisType Basis`), so
`getBasis()` returns a reference into the `DGSoln`, not into a shared singleton.
Binding that to a reference which outlives the owning object is a use-after-free,
and one that can pass for years until an unrelated change moves the allocations.

## Traps worth knowing before you edit

* **autodiff expression templates hold references to their operands.** A lambda
  passed to `trapezoid` (or anything that stores the result) must declare
  `-> Real` explicitly; with a deduced return type it hands back an expression
  referring to dead temporaries, and the symptom is a silently wrong (often zero)
  answer rather than a crash. See `Tests/UnitTests/UtilityTests.cpp`.
* **Never slice an Eigen `solve()` result.** `lu.solve(B)` returns a lazy `Solve<>`
  expression with no coefficient accessor, so `lu.solve(B).topRows(n)` compiles and
  then corrupts the heap; the symptom is a SIGSEGV inside `free()` in an
  unrelated static's destructor. Assign to a `Matrix` first, then slice.
* **Include `<Eigen/Core>` and `<Eigen/Dense>` before the project headers**, the
  way `SystemSolver.hpp` does. The build defines `EIGEN_USE_BLAS`, which swaps in
  BLAS-backed product specialisations; a header that reaches Eigen only through
  `Basis.hpp`'s `<Eigen/LU>` gives its translation units a different set of
  definitions from ones that include `<Eigen/Dense>`. LTO then picks one, and the
  symptom is heap corruption surfacing at exit in whichever static destructor runs
  first — for us, `ChebyshevBasis::singletons`, which the change had nothing to do
  with. `Postprocessing.hpp` carries a comment saying so.
* **`RF_cellwise` and `L_global` hold *time-dependent* boundary data, and only
  `updateBoundaryConditions(t)` may fill them.** `initialiseMatrices` sizes and
  zeroes them; `setInitialConditions` calls `updateBoundaryConditions(t0)` before
  solving the initial `dydt` out of them, and `residual` / `updateMatricesForJacSolve`
  refresh them at their own time. Filled anywhere else, a run with `t0 != 0`
  builds its initial condition from the wrong boundaries, and -- because
  `initialize()` skips `initialiseMatrices` on a reused solver -- a second run
  solves its initial `dydt` out of the previous run's final-time boundary values.
  This is **the `RF_cellwise` trap**, and the general rule: state a run changes
  must be reset in `initialize()`'s unconditional part. Every fixture starts at
  zero, so `the_initial_condition_uses_boundary_data_at_t0` is the only guard on
  the `t0` half.
* **Unanchored `.gitignore` patterns match at every depth.** The root scratch
  entries (`Plots/`, `runs-for-bhavin/`, `scalar-tests/`, `toy-model/`) are
  written with a leading slash for that reason: unanchored, `toy-model/` also
  ignored `python-examples/toy-model/`, so the example was left out of a commit
  with nothing to say so. `git check-ignore -v <path>` names the line
  responsible.
* **Tests reach private `SystemSolver` members** through `MANTA_TEST_PRIVATE`,
  which a `-DTEST` build widens to `public`. No friend declarations needed.
* **Branch on Eigen's major version with `EIGEN_MAJOR_VERSION >= 5`, never
  `EIGEN_VERSION_AT_LEAST`**, whose arguments changed meaning in Eigen 5. Both
  3.4.x and 5.0.x are supported.
* **`extern/autodiff` tracks a fork** (`ianabel/autodiff`, branch
  `eigen-5-singlerange`); `git submodule update --remote` reverts it to an
  upstream that does not build against Eigen 5.
* **With clang, the libstdc++ version is part of the configuration**: no
  `std::bind_front` into `std::function`, and no `std::format` of a whole
  container (libstdc++ 15 only). A build error that splits CI by compiler is
  almost always the standard library. `docs/dev/build.md` has the local
  reproduction, and the g++-14.2 miscompilation behind the gcc 14.3 floor.

## Known limitations

Deliberate and documented; `Tests/README.md` and `TODO` have the full versions.

* **No regression reference carries adjoint output**, so the regression
  driver's adjoint comparison prints `SKIPPING`. `WriteAdjoints = true` writes
  `G`, `G_p` and `z` into the netCDF output, but `AdjointTestProblem.ref.nc` was
  generated without it. The gradients themselves are verified through
  `PyRunner::getAdjointGradients` in `python/Tests/test_adjoint.py` and
  `test_adjoint_aux.py`, and `z` through `test_adjoint_on_demand.py`.
* **A tier-2 field model cannot be written in Python.** `FieldModel` has no
  pybind11 class, so a Python run couples a field's own unknowns only through a
  C++ model named in the dict; `manta.EquilibriumModel` (`PyEquilibriumModel.hpp`)
  is tiers 0 and 1 written in Python. The equilibrium object a Python `Solve`
  returns is held in a `PyEquilibrium`, which takes the GIL to drop it, since
  the last reference may go from a C++ frame.
* **There is no C++ mirror plasma.** `python-physics/mirror-plasma` is the
  implementation, and its own suite is not run by the `python` CTest test
  (`pytest.ini` is `testpaths = python/Tests`). `PhysicsCases/MagneticFields`
  survives for `AdjointPlasma`, with `MagneticFieldTest.cpp` its only coverage.
