# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MaNTA (Maryland Nonlinear Transport Analyzer) solves 1-D reaction–diffusion /
transport systems with a hybridizable discontinuous Galerkin (HDG) spatial
discretisation, integrated in time as an index-1 DAE by SUNDIALS IDA.

`README.md` covers first-time setup (dependencies, `CMakeUserPresets.json`,
installing SUNDIALS). `Tests/README.md` covers test conventions and the current known gaps
in detail — read it before adding tests or interpreting a coverage number.

## Commands

**The build is CMake, out of source.** There is no Makefile in this tree any more
and no `Makefile.local` to write: machine-specific paths go on the configure line
or into a gitignored `CMakeUserPresets.json`.

```sh
cmake --preset default    # configure into build/ (Release; also: debug, coverage, portable)
cmake --build build -j    # solver + libmanta.so + the Python module + the unit tests
ctest --test-dir build    # all three suites

cmake --build build --target MaNTA        # the solver only, at build/MaNTA
cmake --build build --target _manta       # the manta package, python/manta/_manta<suffix>.so
cmake --build build --target UnitTests    # build/Tests/UnitTests/UnitTests
cmake --install build --prefix ...        # headers under include/manta, libmanta.so, manta.pc
pip install .             # the `manta` package and the `manta` console script
pip install .[jax]        # ...and manta.jax (jax, equinox, jaxtyping)

cmake --build build --target unit_tests | regression_tests | python_tests
cmake --build build --target docs         # docs/_build/html, via .venv-docs built from
                                          # docs/requirements.txt; -W, as Read the Docs runs it
cmake --build build --target stubs        # regenerate python/manta/_manta.pyi from the module
cmake --build build --target stubs-check  # fail if the committed stub is stale (CI runs this)
cmake --build build --target typecheck    # mypy over the manta package
cmake --build build --target venv         # .venv from requirements.txt, plus gcovr
cmake --build build --target clean_data   # run output (.nc/.restart.nc/.dat) at the root and in
                                          # Tests/RegressionTests, python/Tests/ and each
                                          # directory under python-examples/ and python-physics/
cmake --build build --target clean_coverage

cmake --preset coverage                            # a build directory of its own
cmake --build build-coverage --target coverage     # all three suites instrumented + gcovr

build/MaNTA --list-options   # every configuration key, straight from ConfigSchema.cpp
```

There is no `clean` target and no need for one: `rm -rf build`.

The regression and Python suites need `requirements.txt` installed, but **nothing
needs to be on `PATH`**. CMake finds one interpreter — preferring a `.venv` in the
repo root — and runs the regression driver and pytest with that one. Name a
different one once, with `-DPython3_EXECUTABLE=...`.

Running one test:

```sh
build/Tests/UnitTests/UnitTests --run_test=solve_jac_tests/solve_hdg_jac_agrees_with_a_dense_solve --log_level=all
build/Tests/UnitTests/UnitTests --run_test=mms_convergence_tests --log_level=message   # see BOOST_TEST_MESSAGE output
pytest python/Tests/test_adjoint.py::test_adjoint_gradient_matches_finite_differences
SOLVER=$PWD/build/MaNTA Tests/RegressionTests/TestSolutions.py --tolerance 1e-2
ctest --test-dir build -R '^unit$' --output-on-failure
```

All three suites run from any working directory. `TestSolutions.py` needs
`SOLVER` when run by hand, because its fallback is `<repo>/MaNTA` and an
out-of-source build does not put one there; CTest sets it. New unit-test `.cpp`
files must be added to `MANTA_TEST_SOURCES` in `Tests/UnitTests/CMakeLists.txt` —
kept an explicit list, unlike `PhysicsCases/*.cpp`, which is globbed with
`CONFIGURE_DEPENDS`.

Build variants are build types and `MANTA_*` options, not command-line variables:
`--preset debug` (`-O0 -g -DDEBUG -DPHYSICS_DEBUG`, and `State.hpp`'s
`checkShapeAndSet` becomes shape-checking rather than a plain assignment),
`--preset coverage`, `MANTA_OPENMP` (threads the cell-independent loops through
`util/ParallelFor.hpp` — see Parallelism below, and read it before turning this
on), `MANTA_VERBOSE`, `MANTA_PHYSICS_DEBUG`,
`MANTA_NATIVE_ARCH`, `MANTA_ASSERTS`, `MANTA_TESTS`, `MANTA_PYTHON`, `MANTA_XLA_FFI`/`MANTA_CUDA`
(JAX FFI, needs jaxlib headers). `cmake -B build -LH` lists them all.

**A Release build does not define `NDEBUG`, deliberately** — CMake would add it
and this build strips it back out, under `MANTA_ASSERTS`, which defaults **on**.
`NDEBUG` disables `assert()` and takes Eigen's assertions with it, and those are
the diagnostic of record here: the adjoint's spatial-parameter transpose was
reported by nothing but Eigen's `resize()` assertion, and under `NDEBUG` it would
have silently transposed a gradient. `MANTA_ASSERTS=OFF` defines it.

**What that costs depends on `k`, and the first measurement of it was taken in
the regime where the answer is "nothing".** Measured 2026-08-25, g++-15.2.0 at
`-O3 -flto -march=native` (znver2), paired A/B, 5 alternating reps, medians,
`MKL_NUM_THREADS=1`:

| workload | asserts on | `NDEBUG` | `NDEBUG` is |
|---|---|---|---|
| `NonlinDiffTest`, 400 cells, k=3 | 5233 ms | 5104 ms | +2.5% |
| `AuxVarTest`, 50 cells, k=8 | 1902 ms | 1740 ms | **+8.5%** |
| `AuxVarTest`, 20 cells, k=10 | 976 ms | 904 ms | **+7.4%** |

So the assertions are close to free at the polynomial degrees the fixtures use
and cost 7-9% at k = 8-10. The mechanism is why: Eigen's assertions guard its
*API* — `operator()`, resize, block construction — not its inner kernels, which
address raw pointers; MaNTA's own are 26 shape checks in `Matrices.cpp`, one or
two per assembled block. Both are O(1) per *call*, so what matters is how many
calls there are per unit of arithmetic, and that is what raising `k` changes: the
blocks get bigger without the call count following.

**The trap here is one of scope, and it is worth remembering rather than the
number.** The first pass measured only k = 3 and k = 4, found −1.6% and +0.3%,
and would have supported "NDEBUG buys nothing in this tree" as a general claim.
It does not; it buys nothing *there*. Any timing statement about this solver that
does not say which `k` it was taken at is under-specified, because `k` moves the
balance between per-cell arithmetic and everything else by more than most changes
do — see the phase breakdown under OpenMP below, where it moves the dominant cost
from a global factorisation to the per-cell loops.

### Where the build lives

| File | What it holds |
|---|---|
| `CMakeLists.txt` | targets, source lists, install rules |
| `CMakePresets.json` | the four presets; `CMakeUserPresets.json` is yours and gitignored |
| `cmake/MantaCompilerFlags.cmake` | warnings, build types, `-march`, the g++-14 warning, `MANTA_GCOV` |
| `cmake/MantaDependencies.cmake` | SUNDIALS, Eigen, netCDF, Boost, BLAS, the `extern/` submodules |
| `cmake/MantaPython.cmake` | which interpreter, and pybind11 |
| `cmake/MantaTools.cmake` | docs, coverage, venv, the two cleaners |
| `Tests/UnitTests/CMakeLists.txt` | `MANTA_TEST_SOURCES`, `-DTEST`, `TEST_DATA_DIR` |
| `python/CMakeLists.txt` | the `_manta` module, XLA FFI, stubs, typecheck |

## Branch protection on `main`

**`main` is protected: work goes on a branch and merges through a pull request.**
Read the live rule rather than trusting this paragraph —
`gh api repos/ianabel/MaNTA/branches/main/protection` — but as it stands:

* **No approving review is required** (`required_approving_review_count: 0`), so
  a PR can be merged by its own author as soon as the checks are green. The PR
  is a gate for CI, not for review.
* **`strict: true`**, so a branch has to be up to date with `main` before it can
  merge. If `main` moves while a PR is open, rebase or merge it in and let CI
  run again.
* **Force-pushes and deletions of `main` are refused**, and there is no linear
  history requirement, so an ordinary merge commit is fine.
* **`enforce_admins` is off**, so the repo owner can still push straight to
  `main`. "Protected" therefore means something different depending on who you
  are, and a workflow that works for `ianabel` is not evidence it works for
  anyone else.

**Nine of the ten contexts `ci.yml` publishes are required**, each pinned to app
15368 (GitHub Actions), so a status of that name from anything else does not
count:

```
Build + tests (g++-15)                    Build + tests (clang++-19)
Build + tests (g++-16)                    Build + tests (clang++-20)
Build + tests (g++-15, Eigen 5.0.1)       Build + tests (clang++-21)
Build + tests (clang++-19, Eigen 5.0.1)   Compile (fedora:latest)
                                          Coverage
```

**The tenth is `Build + tests (g++-15, OpenMP)`, and adding it to the rule has to
wait until the branch carrying it is on `main`.** That ordering is not fussiness;
it is the failure this section already records, approached from the other side. A
required context is matched by *name* against what a PR's own workflow publishes,
and a PR branched from a `main` whose `ci.yml` has no OpenMP leg cannot publish
one. Require it early and every unrelated PR sits at "Expected — waiting for
status to be reported" indefinitely, while the green ticks beside it say the
build is fine. It *was* required early once, in the change that added the leg,
and had to be reverted for exactly that reason.

So the sequence is: merge the leg, then add the context, then check the two agree
with the `diff` below rather than assuming. The g++-16 leg builds
`MANTA_LAPACK=OFF` but is *not* renamed by it, deliberately: it carries no
`label`, so it goes on publishing
`Build + tests (g++-16)` and the required list did not have to move for it. A
matrix key that changes behaviour without changing the rendered name is the
cheap way to add coverage here.

**Those strings are the job's *rendered* name, and that couples the rule to the
matrix.** The job is `name: Build + tests (${{ matrix.label || matrix.cxx }})`,
so adding, removing or relabelling a leg renames its context — and a required
context that nothing publishes is not an error anywhere. GitHub matches exactly;
the check simply never arrives, the PR sits at "Expected — waiting for status to
be reported", and the green ticks beside it are the legs that *are* reporting.
That is not hypothetical: the rule required `Build + C++ tests` — a name no job
has ever published, differing both in wording and in carrying no matrix suffix —
so until it was corrected the protection blocked every non-admin and gated
nothing CI would have caught.

**So whenever a leg is added, dropped or relabelled, update the required list in
the same change**, and check the two agree afterwards rather than assuming:

```sh
gh api repos/ianabel/MaNTA/branches/main/protection/required_status_checks -q '.contexts[]' | sort > /tmp/req
gh pr view <N> --json statusCheckRollup -q '.statusCheckRollup[].name' | sort > /tmp/got
diff /tmp/req /tmp/got     # left-only = required but impossible; right-only = ungated
```

`Coverage` is in the list deliberately. It has no percentage threshold — it runs
the `coverage` target in a `CMAKE_BUILD_TYPE=Coverage` build directory, i.e. all
three suites under an instrumented build, and fails only if the build or a suite
does — so it gates on the same thing the others do
and costs the slowest leg's wall-clock.

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
  sees neither. The docs one *is* caught by the `docs` target, which builds with
  `-W` — but **no CI leg runs it**, so the first thing to notice was a Read the
  Docs build of `main`. Read every resolved hunk that is prose, and run
  `cmake --build build --target docs` before pushing anything touching `docs/`.

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
a_i d_t u_i + d_x sigma_i = S_i(u, q, sigma, phi, x, t)
sigma_i    = sigma_hat_i(u, q, x, t)          # the flux
q_i        = d_x u_i                          # introduced as an unknown
G_j(phi, u, q, sigma, x) = 0                  # nAux algebraic auxiliary constraints
G_s(mu, y, dy/dt, t)     = 0                  # nScalars global (non-spatial) unknowns
R_m(psi, dpsi/dt, y, t)  = 0                  # nField magnetic-field unknowns
```

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
`State::sigmaHat(i)` the physical flux `SigmaFn` returned. Nothing in the tree
reads the incoming sigma at all — `AdjointPlasma` threads it into
`Sn`/`Spi`/`Spe` without using it, and every `dSources_dsigma` is
zero — so a case that starts using it is the first to care, and no test would
catch a sign error there.

### Solve path

`Solver.cpp:runSolver` is the driver. It is three phases, which can also be
called separately:

* `SystemSolver::initialize` — allocate the SUNDIALS objects, build the initial
  condition, open the output files, and *sometimes* run `IDACalcIC`. Two
  independent reasons not to:

  * **A steady solve skips it outright.** `solveSteadyState` drives the whole
    residual to zero from `Y` with KINSOL, so a correction made first is
    discarded by the first accepted continuation step. The gate is
    `solvesForSteadyState()`, so `SteadyStateSolver = "TimeMarch"` and a plain
    transient both keep the call.
  * **A restart skips it too, on the copy path.** It resumes from a state the
    previous run had already driven onto the constraint manifold, and `IDACalcIC`
    cannot find that out cheaply: its convergence test is on the Newton *step*, so
    `IDANewtonIC` calls `lsolve` and only then tests `||J^-1 F|| <= epsNewt`
    (`ida_ic.c:404-417`), `IDAnlsIC` calls `lsetup` unconditionally first
    (`ida_ic.c:345`), and the outer loop runs the whole thing twice on success
    (`ida_ic.c:232`). Measured floor, handed a state it had just converged to:
    **2 residual evaluations, 2 Jacobian builds, 2 Jacobian solves, 0 Newton
    iterations** — every time, and on an already consistent `AuxVarTest` warm
    start at rtol 1e-6 that floor is the whole of the saving: 2 residual
    evaluations of 89 and 2 builds of 21.

    **Only the copy path.** A restart onto a *different* degree is projected —
    `setInitialConditions` transfers `u`, `q`, aux and the scalars and then
    rebuilds `sigma` and the trace — so what it hands IDA is a guess like any
    other, and skipping there is a broken run rather than a saving: `AuxVarTest`
    resuming at a lower degree fails with `IDA_ERR_FAIL` when `IDACalcIC` is
    skipped and completes when it runs. `restartWasProjected` carries that from
    `setInitialConditions` to `initialize`, and
    `only_a_copied_restart_is_treated_as_already_consistent` pins both halves.

  **A cold time-marching run always runs it, and there is no option to turn that
  off.** Its guess is not a consistent state, `IDA_ERR_FAIL` on the first step is
  what starting from one looks like, and a caller who does not care about the
  transient wants `SteadyStateSolver = PseudoTransient` or `Newton` rather than an
  uncorrected time march. `ForceConsistentIC` therefore only ever *adds*
  `IDACalcIC` back — to a steady solve, or to a restart that is not as consistent
  as a restart is supposed to be. `initialize` evaluates the residual once and
  reports its WRMS norm through `getInitialResidualNorm()` on every time-marching
  run, armed or not, so a caller can see how consistent the state they resumed
  from actually was; `initialConditionWasCorrected()` says whether CalcIC ran.

    **The decision is made from what the run *is*, not from a residual threshold,
    and that is a measurement.** This replaced `ConsistentICTolerance`, which
    skipped when the initial weighted residual fell below a number the caller
    supplied. What `IDACalcIC` tests is
    `||J^-1 F||_wrms` — a *correction to y* — and `weightedResidualNorm` says as
    much. The two differ by the per-row amplification `s_i = ||J^-1 e_i||_wrms`,
    and `s_i` is nowhere near proportional to the `ewt` the norm applies. Measured
    per block, as `s_i / ewt_i`, on three cases (`LinearDiffusion`, `MatTest`,
    `AuxVarTest`):

    | block | `s/ewt` | reading |
    |---|---|---|
    | `sigma`, `q` | 0.6 – 10 | about right, and uniform, so harmless |
    | `u` | 2.3e-4 – 2.0 | over-weighted up to ~4000x |
    | `lambda`, Dirichlet ends | **exactly 0** | weight 1e5–1e8 on rows nothing can reach |
    | `aux` | 0.9 – 39 | under-weighted up to ~10x *relative to `sigma`* |

    The `u` rows are the differential ones, whose residual IDA absorbs into `u'`;
    the Dirichlet trace rows are the ones `residual` never writes, so `J^-1 e_i`
    is identically zero there and they can only dilute the mean. The `aux` row's
    spread is the one that bites, because it is a *relative* error against the
    block a corrected state's residual lands in.

    **What that costs, on the tree as it stands, is calibration.** Across six
    `AuxVarTest` warm-start states — three tolerances, corrected and not — the
    amplification `||J^-1 F|| / ||F||` runs from **15 to 187**. One problem, one
    discretisation. A threshold on `||F||` therefore means something different at
    each of them, which is what makes "set it per problem" honest advice rather
    than a hedge.

    **Until recently it was worse than uncalibrated — it was inverted**, and that
    is how the underlying defect was found. Before `AuxVarTest`'s missing
    `dSigma_dPhi` block was declared, a warm start there measured `||F||` = 1.6e-4
    uncorrected against 3.8e-4 corrected — the uncorrected state 2.4x *better* —
    while `||J^-1 F||` made it 2.0e-2 against 3.1e-3, 6.3x *worse*. The run sided
    with the second: skipping failed with `IDA_CONV_FAIL`, correcting worked. IDA's
    own log showed why — the failing Newton's correction plateaued at 1.98e-2 as
    `h` fell, which *is* `||J^-1 F||`, while `||F||` could not see it. Note that
    `||J^-1 F||` predicted that failure using the *defective* `J`, which is the
    point of the quantity: it measures the Newton the solver will actually run,
    not the one it ought to.

    That fixture is fixed. `SigmaFn` adds `(a - u*u)` to both variables' fluxes
    while the derivative was declared for variable 0 only, so off the manifold —
    i.e. exactly on a warm start — Newton diverged; finite-differencing put the
    block out by 98% before and 4.2e-9 after. **Every restart round trip in the
    suite now completes with the skip armed**, at every tolerance measured, and on
    the current tree the two norms order those states the same way. So there is no
    longer a case in the tree that *requires* `IDACalcIC`.

    So there is no number to pick, which is why the decision is now made from what
    the run *is*. A test that would be correct — `||J^-1 F||` itself — costs one
    residual, one Jacobian build and one Jacobian solve against `IDACalcIC`'s floor
    of two of each: a factor of two, not a near-free test, and not obviously worth
    having when the run already knows whether it is a steady solve or a copied
    restart.

    ~~Note the opposite failure too: a TestDiffusion warm start at rtol 1e-6 /
    atol 1e-8 cannot complete `IDACalcIC` at all.~~ **That was a MaNTA bug, and it
    is fixed.** `initialize` passed `IDACalcIC` the *interval* `dt0 > 0 ? dt0 : dt`
    where `tout1` wants an absolute time. Every fixture in the tree starts at
    `t0 = 0`, where the two agree bit for bit, so no cold start ever noticed; a
    restart resumes at the time the file was written, and that fixture restarts at
    `t0 = 0.05` with an output cadence of `0.05`, so `tout1` came out *exactly*
    equal to `t0` and IDA refused the input — `IDA_ILL_INPUT` (-22), before
    evaluating a single residual. Nothing to do with the tolerance: it reproduced
    at every tolerance, and with `tout1 = t0 + dt` that same warm start converges
    in 3 residual evaluations, as does a degree-projection restart whose weighted
    residual is 8.7e3. Worse was available — a restart with `dt0` set and
    `t0 > dt0` handed IDA a `tout1` *behind* `t0`, i.e. the wrong direction of
    integration. For scale, TestDiffusion round trips separate cleanly (cold 0.30
    at `atol = 1e-3` and 417 at `1e-8`; warm 7.7e-4 to 1.9e-2).

  On either skip the `t0` output slice sees the guess `setInitialConditions`
  built, which is the state the run really started from.

* `SystemSolver::integrate(tFinal)` — the time loop, then the adjoint solve and
  the final netCDF / restart output.
* `SystemSolver::destroySundials` — free all of it. Idempotent, and safe with no
  preceding `initialize`, which is what lets `runSolver` free on both the normal
  and the exceptional path.

**A steady solve can also be taken in slices.** `MaxContinuationSteps` (default
200) bounds one `solveSteadyState`; running out of it is a budget exhaustion, not
a failure of method, and `continueSteadyState()` resumes from the state *and* the
pseudo-time step SER climbed to. `integrate()`'s tail is factored as
`writeSteadyState()` + `finishRun()` so a sliced solve ends the same way an
unsliced one does; `finishRun` is shared with the time-marching branch and closes
the output files, so it runs once per run. A second `solveSteadyState()` would resume from
neither — `SteadyState.cpp` re-enters at `PseudoTransientInitialStep` unless the
`resume` flag says otherwise — and re-climbing the ramp is the whole solve rather
than a margin on it: a `NonlinearDiffusion` needing 15 continuation steps takes
the same 15 in slices of three when each resumes, and does not converge in 40
slices when each starts over
(`a_resumed_steady_solve_does_not_re_climb_the_ser_ramp`). Slicing requires
driving the phases directly, because `runSolver` frees the state on its way out
of a failed solve, so `PyRunner::run_ss()` cannot do it.

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
SUNLOGGER_INFO_FILENAME=/tmp/ida-info.log SUNLOGGER_DEBUG_FILENAME=/tmp/ida-debug.log build/MaNTA foo.conf
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
* **`IDA_ILL_INPUT` (-22) with "tout1 too close to t0" is not about the state at
  all** — it is `initialize` handing `IDACalcIC` the wrong `tout1`. That argument
  is an absolute *time*, "the first value of t at which a solution will be
  requested", and it used to be passed the *interval* `dt0 > 0 ? dt0 : dt`. The
  two agree only at `t0 = 0`, which is where every fixture in the tree starts, so
  `t_initial = delta_t` would make `tout1` land exactly on `t0` and kill the run,
  so it is `t0 + (dt0 > 0 ? dt0 : dt)` — the first time `integrate()` asks for —
  and `initialize_starts_at_a_nonzero_time` pins that. Worth knowing because the
  symptom *looks* like a hard initial condition and is not: it reproduces at every
  tolerance, and a warm start that hits it converges in 3 residual evaluations
  once `tout1` is right.
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

### Parallelism (`MANTA_OPENMP`, off by default)

**Every `#pragma omp` in the tree is inside `util/ParallelFor.hpp`.** Nothing
else may write one. `manta::parallel_for(n, body, grain)` is the only entry
point, and it exists because three properties were needed at all eight sites and
present at none of them.

**Exceptions.** A physics hook throwing is *supported*: `static_residual`
catches it, prints, and returns 1, which IDA treats as recoverable and retries
with a smaller step. It is also how a Python case's exception reaches the
solver. An exception that escapes an OpenMP structured block does not
propagate — gcc's outlined function has no handler above it, so
`__cxa_call_terminate` aborts the process. With the bare pragmas this replaced,
`MANTA_OPENMP=ON` **aborted the unit suite**:

```
__cxa_call_terminate / __cxa_throw
ThrowingDiffusion::SigmaFn(...)
TransportSystem::SigmaFn(...) [clone ._omp_fn.0]
libgomp.so.1
```

Note how it hid: run that test alone and the throwing iteration lands on the
master thread, where the exception *can* reach the handler, and it passes. It
takes a worker thread to kill the process. `parallel_for` catches per iteration,
keeps the first exception, and rethrows on the calling thread.
`utility_tests/an_exception_from_the_body_reaches_the_caller` pins it, and throws
from the *last* index for that reason.

**A trip-count floor**, because forking a team for a handful of iterations costs
more than it saves and the fixtures here are 3–10 cells. The two grains differ by
16x on purpose: `TransportSystem::physicsGrain` is 64 because an iteration is one
pointwise hook call, `SystemSolver::cellGrain` is 4 because an iteration is a
dense factorisation. A single value cannot serve both — a floor of 32 on the cell
loops would have turned off the *only* regime where threading actually pays.

**Only the solver's cell loops are threaded. The physics is never threaded, and
that is a policy rather than an omission.** The default batched wrappers in
`TransportSystem.hpp` — the fallback for a case that supplies only pointwise
hooks — are plain serial loops. A case that has not provided a batched
implementation is assumed to have a reason, and threading its hooks would call
arbitrary case code concurrently on one instance, which nothing here can check
and the case never agreed to. A case that *wants* parallel physics overrides the
batched level, which is what those methods are virtual for.

For a Python case the rule is not merely prudent but load-bearing. Every
pointwise trampoline in `PyTransportSystem` takes the GIL, so N threads serialise
on it and pay a lock handoff *per point*. Measured: with the wrappers threaded,
`MANTA_OPENMP=ON` at four threads took the Python suite from ~110 s to **over
1500 s, where it hit ctest's timeout without finishing** — a floor of 13.6x
slower, not a failure to speed up. It reached CI as a red required leg. With the
wrappers serial the same suite is **109.6 s at `OMP_NUM_THREADS=4`**, i.e. back
to indistinguishable from serial. The vectorised path was always right:
`PyTransportSystem::ComputePhysics` takes the GIL *once* for the whole grid, and
`manta.jax` goes through it.

Parallel, then: the per-cell factorisation in `updateMatricesForJacSolve`, the
per-cell solves in `solveHDGJac`, and both back-substitutions. Serial, each
saying so at the site: anything accumulating into `K_global` or `F`, because
**lambda lives on cell faces** so neighbours share the block they write; and the
adjoint's matrix build, which grows `adjoint_CEBlocks`/`adjoint_CGBlocks` with
`emplace_back` and runs once per run rather than once per Newton iteration.

**Nothing under a parallel loop calls into a physics case any more, and getting
there found real waste.** `assembleCellMatrix` — which *is* inside the threaded
cell loop — built its `X` block by integrating `alphaValue * aFn` through a
30-point Gauss rule per entry, per variable, per cell, on every Jacobian build.
`aFn` is a pointwise hook and `PyTransportSystem` overrides it with
`PYBIND11_OVERRIDE`, so a Python case's `aFn` was being called from OpenMP worker
threads, taking the GIL per quadrature point. But `initialiseMatrices` already
stores exactly that matrix unweighted in `XMats`, `MassMatrix` is linear in its
weight, and `aFn` takes no time argument — so `alphaValue * XMats[i]` is the same
quantity for none of the work. Removing the recomputation is worth **24-32%** on
its own:

| | before | after |
|---|---|---|
| k=3, 400 cells | 555 ms | 395 ms |
| k=4, 200 cells | 2147 ms | 1624 ms |
| k=8, 50 cells | 1226 ms | 828 ms |
| k=10, 20 cells | 755 ms | 512 ms |

The general point is worth more than the number: a quantity that depends on
neither the state nor the time was being rebuilt once per Newton iteration
because the assembly that needed it was written in terms of the hook rather than
in terms of the thing already derived from the hook.

**What threading is worth**, with all of the above in place. `MKL_NUM_THREADS=1`,
best of 3:

| | 1 thread | 4 threads |
|---|---|---|
| k=3, 400 cells | 482 ms | 371 ms (1.30x) |
| k=4, 200 cells | 2007 ms | 1235 ms (1.63x) |
| k=8, 50 cells | 1183 ms | 687 ms (1.72x) |
| k=10, 20 cells | 676 ms | **379 ms (1.78x)** |

Note that taking the physics *out* of the parallel region cost nothing and helped
everywhere: every absolute time above improved, the single-thread baselines
included, because `parallel_for` was charging its own overhead — an atomic flag, a
try/catch, an `exception_ptr` — on loops whose bodies were a single hook call.

**Threading changes no answers, and that is checkable rather than hoped for.**
There is no reduction anywhere in `parallel_for` — every iteration writes its own
slot — so the arithmetic is the same operations in the same order whatever the
team size. Verified the way `CLAUDE.md` verifies this class of claim elsewhere:
the k=10 case run by the serial build and by the OpenMP build at four threads
gives **byte-identical `.nc` and `.restart.nc`**, not merely agreement at the
regression suite's 5e-3. Re-run that after touching `util/ParallelFor.hpp` or any
loop that goes through it; the regression tolerance is far too loose to see a
change of this kind. It is also the sharp contrast with BLAS threading below,
which *does* move the last bits.

Both suites pass under `MANTA_OPENMP=ON` at `OMP_NUM_THREADS=6`, and the unit
suite is *faster* that way — 16.1 s against 26.5 s serial, because
`MMSConvergenceTests` runs at exactly the high degrees where this pays.

**Oversubscription is not a mild loss here — it is a 2x regression.** Eight
threads on 20 cells costs 1963 ms against 528 at four, and eight on 200 cells
costs 33359 against 16399 at one. Cap `OMP_NUM_THREADS` near the cell count and
well below the core count; there is no `num_threads` clause in `parallel_for`
doing it for you.

**Building with OpenMP silently turns on BLAS threading, and that changes
answers.** `-fopenmp` loads `libgomp`, which is enough for a dispatching BLAS to
start threading itself — on the development box `libblas.so.3` is
`libmkl_rt.so`, and with `OMP_NUM_THREADS=6` the multithreaded MKL changed
`dgemm` reduction order enough that `afn_tests/the_jacobian_agrees_with_the_
residual_for_a_nonunit_coefficient` failed with `IDACalcIC could not complete`.
**That test parallelises none of MaNTA's own loops** — `nCells = 3` is below
`cellGrain` and its 12 physics points are below `physicsGrain` — which is what
makes the attribution certain, and it was confirmed by separating the two
variables: `OMP_NUM_THREADS=6 MKL_NUM_THREADS=1` passes, `OMP_NUM_THREADS=1
MKL_NUM_THREADS=4` fails. Set `MKL_NUM_THREADS` (or `OPENBLAS_NUM_THREADS`)
explicitly whenever `OMP_NUM_THREADS` is set, and note the two want opposite
things: BLAS threading is the *only* thing that helps the k=4 case (1.39x) and it
hurts the k=10 one (1.7x slower).

**Nothing in CI builds with `MANTA_OPENMP=ON`**, which is why all of the above
survived to be found by hand. The `utility_tests` cases are the guard, but they
only bite in a build that sets it.

### The trace solve (`util/BandedMatrix.hpp`)

Static condensation leaves a matrix on the cell faces, `K_global`, and
`solveHDGJac` factorises it on **every Newton iteration**. It used to be a dense
`Eigen::FullPivLU` of side `nVars * (nCells + 1)` — O(nCells^3) in the one
quantity the method exists to make O(nCells) — and it was **91% of a 400-cell
k=3 run and 73% of a 200-cell k=4 one**. It is now a banded LU with partial
pivoting: `dgbtrf`/`dgbtrs` when a LAPACK was found, an equivalent built-in when
not. Measured 9.4x and 7.6x on those two cases; `TODO` has the table and the
per-phase numbers behind it.

**It is banded only in an ordering the solution vector does not use.** Lambda is
laid out `var * (nCells + 1) + node`, and in that order two nodes of different
variables sit `nCells + 1` apart — a full-width matrix. Indexed `(node, var)`
instead, cell `i` touches only nodes `i` and `i+1`, so the bandwidth is
`2 * nVars - 1` either side. The band form therefore carries **its own ordering**
and the solve gathers into it and scatters back (`toTraceMajor` /
`fromTraceMajor`). That is deliberate and worth preserving: the DOF layout, the
restart format, `DGSoln::Map` and the pybind11 casters are all untouched, and the
permutation is O(n) against a solve that is now O(n * band^2).

**`K_global` is singular, and the dense decomposition was hiding it.** A
Dirichlet end sets `Hvar`'s diagonal to zero (`initialiseMatrices`) and nothing
else writes that trace DOF, so the row *and* the column are identically zero —
the same rank deficiency `CLAUDE.md` already records for the finite-differenced
Jacobian, here in the operator itself. `FullPivLU` returns the particular
solution with the free components zeroed, which is the right answer arrived at by
accident; a banded LU has no such behaviour to fall back on and reports the zero
pivot. So `imposeDirichletTraceRows` writes the constraint down — identity on the
row, zero on the right-hand side, which is correct because `delta lambda = 0` at
a face already sitting at `g_D(t)`. Measured rank 3/5 and 4/6 on the fixture
solves, with the banded answer matching the dense one to 2.7e-15. **If you touch
the boundary assembly, that identity is load-bearing**: without it the solver
throws on the first step rather than degrading quietly, which is the one mercy
here.

**The singular-matrix report is a `throw`, and it needed a catch that was not
there.** `SunLinSolWrapper::Solve` is a SUNDIALS C callback — it reaches
`solveJacEq` through a function pointer — so an escaping exception is undefined
behaviour, exactly the hazard `static_residual` was written to close. It now has
the same try/catch, returning 1, which IDA treats as a recoverable linear-solver
failure and responds to by cutting the step and re-forming the Jacobian. Worth
knowing that this closes a **pre-existing** hole too: `solveJacEq` allocates, so a
`bad_alloc` — or anything a field model threw — was already unwinding through C
frames before any of this.

**This changes answers at round-off, unlike the OpenMP work.** A different
factorisation moves the last bits and IDA's step sequence follows, so byte
comparison is the wrong check — measured worst relative difference in the netCDF
output is 4.2e-13 at k=3 and 2.7e-11 at k=10, against run tolerances of 1e-8 and
1e-6. Contrast the threading, which is byte-identical because it is the same
operations in the same order. Know which kind of change you are making before
choosing the check.

**LAPACK is optional and pinned to the BLAS's vendor, and the reason is the
dlopen trap.** `cmake/MantaDependencies.cmake` asks `FindLAPACK` for whatever
vendor the BLAS actually resolved to, copies `LAPACK_LIBRARIES` **out by value**,
and links the paths rather than `LAPACK::LAPACK`. That last part was measured, not
guessed: CMake's `FindLAPACK` creates the imported target under
`if(NOT TARGET LAPACK::LAPACK)` but sets `INTERFACE_LINK_LIBRARIES` on it
*outside* that guard, so `SUNDIALSConfig.cmake`'s `find_dependency(LAPACK)` a few
lines later **rewrites the contents of the target this project already linked**.
On the development box that put `libmkl_gf_lp64 + libmkl_gnu_thread +
libmkl_core + libgomp` — the layered link that is unsafe to `dlopen`, and the one
the BLAS block exists to avoid — onto the link line of `libmanta` and the *Python
module*, while an isolated `find_package(LAPACK)` with the same `BLA_VENDOR`
resolved cleanly. It is the BLAS trap one level deeper: not the variable this
time, the target.

The block also `unset`s `LAPACK_LIBRARIES` from the cache before every find. That
is what makes it self-healing: `FindLAPACK` short-circuits on a cached value, and
without the unset a single bad configure is permanent — the poisoned value is
read back, captured and re-pinned forever, and reconfiguring does not clear it.
Verified by poisoning a build directory and watching it recover.

**LAPACK is not faster than the built-in here, and that is expected rather than
disappointing.** Measured within noise on all four benchmarks, the built-in
marginally ahead on three (537 vs 555 ms, 1180 vs 1226, 718 vs 755, 2163 vs
2147). The bands are 1 wide for a single-variable case and 3 for two, so
`dgbtrf`'s blocking has nothing to exploit and its call overhead is comparable to
the arithmetic. LAPACK is preferred because it is the reference implementation
and someone else maintains it — not for speed — and it will start to matter for a
case with many variables, where the band is `2 * nVars - 1`. So do not "optimise"
by dropping the LAPACK path on the strength of these numbers; they say the two
are equivalent at `nVars` of 1 or 2 and nothing about `nVars` of 8.

`utility_tests` exercises **both** the LAPACK path and the built-in one in every
build, whatever the build found, through `factorizeBuiltin`/`solveInPlaceBuiltin`.
Without that the fallback would rot on every box that has LAPACK, which is most
of them — and it is the boxes that do not that need it.

### Configuration

**Every key MaNTA accepts is declared once**, in `ConfigSchema.cpp`: canonical
name, deprecated aliases, type, category, per-reader requiredness, default and a
line of documentation. `build/MaNTA --list-options` prints the table. There used to
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

### Self-consistent magnetic fields (`FieldModel`)

A `FieldModel` (`FieldModel.hpp`) contributes `nFieldDOF` unknowns `psi`, one
residual row each, and `nGeometry` derived *geometry slots* `g_s(psi, x, t)`.
It follows the physics-case pattern throughout — a validated spec, a
process-global registry with the same two throws, its own toml table, selected
by name from the config. `docs/field_coupling.rst` is the interface document and
`--list-options` prints the five keys; what follows is what neither says.
`docs/superpowers/notes/2026-08-16-self-consistent-b-fields-design-notes.md`
records why the design has this shape, where it departed from its own spec, and
every number quoted below.

**The geometry slots are the only channel into the transport physics**, and they
are not unknowns: they are a function of `(psi, x)` evaluated at the physics
nodes and cached per residual, in the same standing as `sigmaHat`. A case reads
them as `State::geom(s)`. Nothing else crosses in either direction.

**`FieldModelSpec` is not `FieldSpec`.** `SystemSpec.hpp` already defines a
global `struct FieldSpec`, the per-transport-variable descriptor bound to Python
as `manta.Field`. Reusing the name compiles every translation unit cleanly and
fails only at link time, as an ODR violation naming neither type. Do not shorten
it back.

**The whole thing is inert when unused, and that is checked by hand rather than
asserted.** `psi` goes *last* in the layout so nothing before it moves, and
every existing config has no field model. The check is to build `main`, run both
binaries over every `Tests/RegressionTests/*.conf` from the same directory and
`cmp` — netCDF files carry no timestamp of their own, so a byte comparison is
legitimate, and the regression suite's 5e-3 is far too loose to see a change of
this kind. Last measured: all 14 `.nc` byte identical; all 14 `.restart.nc`
identical apart from the deliberately added `int nField = 0`. Re-run it after
anything that touches the DOF layout, the residual or the Jacobian solve.

**Which test catches which failure class — the three-way split is the point.**

* A wrong `A1` or `A2` (the coupling Jacobian blocks) costs Newton iterations
  and *nothing else*, because the coupled Jacobian is never assembled. Only
  `FieldJacobianTests.cpp` sees it, by finite-differencing the residual and
  requiring `J dy = g` with `FieldSolve = exact`.
* A wrong coupled *residual* — a sign, a factor — converges at the right rate to
  the wrong function. No Jacobian check sees it; only the closed-form comparison
  in `MMSFieldTests.cpp` does. A 5% error in the field row, not even a sign
  error, drops the `k = 2` orders to 0.26, 0.00, -0.00.
* A wrong *transpose* of either is a silently wrong gradient beside a perfectly
  good `G`, and only `FieldAdjointTests.cpp` sees that. **This is the adjoint
  asymmetry, and it is the same one `dSigma/dPhi` demonstrated**:
  `initializeMatricesForAdjointSolve` stores `A1^T` and `A2^T` beside `M^T`, so
  a coupling added to `updateMatricesForJacSolve` and not to it degrades a
  forward run's convergence and corrupts a gradient. They are *materialised*
  rather than transposed at each use precisely so a test can zero one and
  require the gradient to go wrong.

Two more properties nothing else pins. `resetForRun()` is called from the
**unconditional** part of `initialize()` — not from `initialiseMatrices`, which
`initialize` skips when already initialised: that is the `RF_cellwise` trap, and
a model caching an equilibrium across runs is exactly the shape that falls into
it. `a_coupled_solver_reused_matches_a_fresh_one_bit_for_bit` is the only thing
standing between that and a second run that completes, looks plausible and is
wrong; breaking either the fixture's `resetForRun` or `initialize`'s call to it
moves the answer by 3.2e-4. And `psi_round_trips_through_a_restart` uses a
**differential** field DOF deliberately: `IDA_YA_YDP_INIT` solves for algebraic
*values*, so an algebraic `psi` would be recomputed from the restored transport
state and the case would pass without ever reading `psi` off disk.

**`FieldSolve = iterative` is a cost choice, never an accuracy one.** The block
Gauss-Seidel sweep escalates to the exact Schur solve when it exhausts its cap,
in *both* the forward and the adjoint directions, so it can be slower than
`exact` and can never be less accurate. The break-even is
`#sweeps < nField + 1` — one transport solve per sweep against `nField + 1` for
the exact solve — and **no fixture in this tree is on the winning side of it**:
iterative is ~1.5x more expensive at `nField = 1` and 2.2-6.3x at `nField = 5`.
It is a bet on `N_magnetics >> N_HDG`, which nothing here exercises. Note that
isolated Jacobian solves with *random* right-hand sides needed 13-38 sweeps at
`nField = 5`, while a real integration averages 2.5-3.6 and has never hit the
cap; Newton's right-hand sides are far more benign than random vectors, and
neither number says anything about the cap on its own.
`FieldSolveMaxAdjointSweeps` defaults to 100 against `FieldSolveMaxSweeps`'s 20
because the adjoint always runs at `cj = 0`, where the coupling is stiffest.
`TODO` records two candidate latches for a run that falls back on every solve;
neither is implemented.

The refusals, each at the earliest point the combination is known:
`nScalars > 0` with a field model (`setFieldModel` — the non-superconvergent
`dSources_dScalars` branch builds its `State` from `DGSoln::evalOnNode`, which
has no geometry rows, so a case reading geometry there would work with
`Superconvergent = true` and read out of bounds with it off); a field DOF
declared differential whose row carries no `d/dt` (`initialize`, naming the DOF,
because left to IDA it is `IDA_LINESEARCH_FAIL`); a restart whose file's
`nField` disagrees with the configured model's; and **four naming refusals that
exist because the spec's names are now netCDF names** — a name netCDF would
reject, and a DOF sharing a name with a geometry slot, both in
`FieldModelSpec::validate()`; a group name colliding with a transport variable,
an aux variable or one of `Grid`/`RestartData`/`x`/`t`/`nVariables`, in
`setFieldModel`, which is the earliest point that knows both. Left to netCDF
these are an `NcBadName` or `NcNameInUse` out of `ncGroup.cpp` at the *first
write*, naming netCDF's source and a line number and neither MaNTA nor the
spec. The DOF-versus-slot one is new with this layout: the two lists used to be
written nowhere near each other and now share one group, and `checkNames`
compares each list only with itself.

**The restart test's oracle is the raw netCDF array, not `getSolution()`.**
`yJac` is filled through `DGSoln::copy`, which is the function that carries the
field block — so a comparison rooted in `getSolution()` on both sides agrees at
zero when `psi_ = other.psi_` is deleted, and the case passes while `psi` is not
being copied at all. Measured: green under that deletion with the old oracle,
three failures with the new one. The same trap applies to any future check of a
quantity `copy` moves.

### Python layer

`Python.cpp` defines `_manta`, the compiled core of the **`manta` package**;
`python/manta/__init__.py` re-exports it and adds the parts better written in
Python. Users `import manta`, and `pip install .` makes that work from anywhere
(the build shells out to CMake, reusing `build/` if it is configured, so it still
needs a working dependency setup — see `setup.py`).
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

  **A steady solve can be driven in slices from here too**, which is what makes
  the resumable continuation above reachable from Python: `start_steady` /
  `continue_steady` / `finish_steady` / `abandon_steady`, wrapped as
  `manta.SteadySolve`. Three things are load-bearing. `OutOfSteps` is *returned*
  and a `SolverFailed` throws, so a driver tells a spent budget from a dead solve
  without reading a message — which relies on `solveSteadyState` clearing
  `steadyOutcome` on entry, since `finish()` is the only thing that sets it and
  an exception from inside the residual bypasses it entirely; a stale
  `OutOfSteps` there is an infinite loop, not a wrong label. Each slice calls
  `captureState()`, because
  `getSolution` reads `yJac` and would otherwise hand back the initial condition
  — silently, since `yJac` is always *a* valid state. And a live loop owns
  SUNDIALS objects that **nothing else frees** — `~SystemSolver` does not call
  `destroySundials` — so `configure()`, `~PyRunner` and any exception out of the
  loop all abandon it explicitly; that is why the context manager is the form to
  prefer. `DegreeAdaptation` is refused, since adapting the degree replaces the
  solver the loop is holding.
  The same four names are FFI ops (`ffi.hpp`, CPU only like `run`/`run_ss`), so
  `manta.SteadySolve(ffi_runner)` works — `steadyStats()` and
  `objectiveEstimate()` need none, being host-side reads that touch no device
  memory. The outcome crosses as a concrete `int32`, which forces the sync a
  Python `while` needs, so a slice loop belongs in eager code or inside an
  `io_callback` rather than under `jit`.

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
the `python` CTest test — `pytest.ini` is `testpaths = python/Tests` — so nothing
in CI runs them, and their READMEs carry the status instead.

### Type stubs

`python/manta/_manta.pyi` is **generated** by the `stubs` target; `stubs-check`
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

`cmake --install build --prefix ...` installs the headers under
`$PREFIX/include/manta`, `libmanta.so`, and `manta.pc`. **`manta.pc` is written at
*install* time, not configure time**, so `--prefix` is recorded correctly and
`DESTDIR` is honoured; a plain `configure_file` bakes in whatever the prefix was
when cmake last ran, and the first symptom of that is a plugin compiling against
the wrong headers. There is deliberately **no installed CMake package**: an
exported target would invite `target_link_libraries(mycase manta)`, and linking
`libmanta` is exactly the mistake below. A physics case built as a shared object and named
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
`dg/dphi` from the batched `dg` and weights it. A C++ case's `dgFn_dphi` still
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
guard that the two operators still differ on it. Those two guards are the whole of
the coverage, so keep them: a reference built "straight from the weights" pins the
formula, not the operator, if the data cannot tell them apart.

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
* **Build staleness bit three times under the Makefile**, each time because a
  hand-maintained prerequisite list had a hole in it. CMake derives the whole
  dependency graph, so that class is gone — with one exception, and it is the one
  the port itself tripped over. **`python/manta/_manta<abi>.so` is written into
  the *source* tree**, because that is where `import manta` has to find it, so it
  is the one output two build directories can fight over. `build/` and
  `build-coverage/` both target it, and CMake will consider its own target up to
  date if the file on disk is newer than its objects — even when the file was put
  there by the other build. The symptom is a test suite exercising a module you
  did not build: measured, a Release run reporting a crash that belonged to the
  instrumented module.

  **Each build directory now claims the module, so this is handled rather than
  remembered.** `manta_claim_module` (`python/CMakeLists.txt`) is an ordering
  dependency of `_manta`: before any link it checks the file against
  `<build>/python/manta_module.stamp`, which a POST_BUILD step wrote describing
  what this directory last linked, and deletes anything it does not recognise —
  which is what makes the link happen. `cmake/MantaClaimModule.cmake` is both
  halves. Building the same directory twice does no extra work; a relink happens
  only when the module really did belong to someone else.

  Both directions were measured, and **both were silent**. A coverage run whose
  Python suite imported the *Release* module: 133s against 748s for the same
  tests, `PyRunner.cpp.gcda` left at the previous run's timestamp, and the report
  still read correctly only because gcov data accumulates and an earlier
  instrumented run had left some behind. And `cmake --build build --target
  _manta` reporting `Built target _manta` while leaving the 57MB instrumented
  module exactly where it was — note what that means: **"rebuild it in `build/`"
  was never a workaround**, because the rebuild is precisely the thing that does
  not happen. The docs said it for a while; it was wrong.

  On top of that, `cmake/MantaCheckInstrumented.cmake` runs ahead of the suites
  in the `coverage` target and refuses to start unless the module carries `.gcda`
  strings. That is deliberate belt and braces: the claim is a mechanism that
  could quietly stop working, and the thing it protects — a coverage number —
  looks equally plausible either way.

  Two details worth keeping if you edit any of it. The claim has to hang off
  `_manta` rather than off the targets that *use* the module; an
  `add_dependencies` on `coverage` or on a test would be too late twice over,
  since that dependency is satisfied before the target's own commands run and is
  satisfied by exactly the stale comparison at fault. And the path cannot be
  spelled `$<TARGET_FILE:_manta>` — a `TARGET_FILE` genex in a custom command
  makes that command depend on the target, and this command is a dependency of
  it, so CMake refuses the cycle by name. It is assembled from
  `OUTPUT_NAME`/`PREFIX`/`SUFFIX`, with a configure-time check that the suffix
  still looks like a module suffix, because a mis-assembled path would delete
  nothing and restore the original silent bug.

* **g++-14 miscompiles this tree at `-O3 -flto -march=native`, and the symptom is
  a wrong number rather than a crash. Do not trust a g++-14 release build.**
  Measured 2026-08-23 on g++-14.2.0, znver2. It is no longer a CI leg, and the
  README's floor for gcc moved to 15 because of it.

  The trigger is **any change to `SystemSolver`'s member layout**. Adding one
  inert member — a `bool` and an unused `std::vector<double>`, referenced by no
  code anywhere — took a clean tree from 12/12 passing to 8/12 failing, on a test
  that densely finite-differenced `residual()` inside the test translation unit.
  Only the `AuxDiffusion` cases failed, plain and superconvergent, at every `k`,
  with a relative drift of about 1.24: an O(1) error, not a tolerance one.
  Nothing else in the suite ever failed.

  **Reproducing it needs a test of that shape**, and the tree currently has none
  wired for it; `SolveJacTests.cpp` differences the residual the same way and is
  where to build one. The two ingredients are a dense finite-difference of
  `residual()` inlined into a test TU and a change to `SystemSolver`'s member
  layout. The defect was never diagnosed, only bounded, so treat it as live.

  What breaks is the **finite-difference reference**, not the assembly. `|J|` of
  the assembled Jacobian is bit-identical every run (7.9144520420784605 at
  k = 1); the differenced Jacobian loses the column of the first interior trace
  DOF — index 33 at k = 1, 49 at k = 2, 65 at k = 3, always `lambda[1]` —
  differencing it to 0 against an assembled -1.5. The drift is then exactly
  1.5 / 1.2071067811865475 = 1.2426406871192851, which is how you recognise it.

  **All three of g++-14, `-flto`, and `-march=native` are required.** Drop any
  one and it is 10/10 clean; so is `-O0`. g++-15 is clean, clang++-19 is clean.

  Ruled out, each by measurement rather than by argument: build staleness;
  leftover output files; ASLR (`setarch -R` still flakes, and with a fixed
  environment, which is the odd part — passing runs are bit-identical to each
  other while failing runs all differ); BLAS threading; uninitialised trivial
  automatics (`-ftrivial-auto-var-init=zero` *and* `=pattern` both still flake);
  and anything AddressSanitizer sees — under ASan it is 12/12 clean with no
  invalid access reported.

  **It is a heisenbug, and that is what makes the usual bisection useless.** Any
  instrumentation in the affected translation unit makes it vanish: a read of `Y`
  before the differencing, one extra term in an existing `BOOST_TEST_MESSAGE`, a
  store into a file-scope array. So do `-fno-strict-aliasing` and
  `-fno-tree-vectorize` — and **that is the trap**: with *every* codegen
  perturbation hiding it, "flag X makes it pass" carries almost no information,
  and reading `-fno-strict-aliasing` as evidence of an aliasing violation would
  be reading noise. The only reliable signal is the one asymmetry: adding a
  member to `SystemSolver` reliably *creates* it. Change what is computed, or
  change the compiler; do not try to print your way to it.

  **Scope, as measured.** `MaNTA` itself — the solver binary, built without
  `-DTEST` from a different object set — produced **bit-identical netCDF output
  over 8 runs** of `Tests/RegressionTests/AuxVarTest.conf`, the aux-variable
  fixture, with the inert member in place. So the run-to-run variation is
  confined to the unit-test binary, which points at the header-only
  `fdjac::jacobian` being inlined into the test TU rather than at the solver. That
  bounds it; it does not clear it, because a deterministic wrong answer is still
  wrong and nothing here has checked the solver's numbers against anything but
  themselves.

  **g++-14 has been dropped from CI entirely** — both build legs and the coverage
  job, which moved to g++-15 even though at `-O0 --coverage` it was never exposed.
  The Eigen 5.0.1 leg moved with them and became two, on g++-15 and clang++-19.
  Count the cost honestly: g++-14 is what Ubuntu noble's archive ships, so the
  gcc most people have by default is now the one this project tests least. The
  release build warns when it sees it; `TODO` has the full reproduction.

* **gcc and clang do not diagnose the same things, so build with clang
  occasionally** — that is what CI's clang matrix legs are for. (CI is seven
  `build-and-test` legs: g++-15/16 and clang++-19/20/21 against the distro's
  Eigen 3.4.0, plus g++-15 and clang++-19 against Eigen 5.0.1; then a Fedora container job that
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
* **Third-party includes must be `SYSTEM`.** An imported target's include
  directories are already treated that way; anything added by hand needs
  `target_include_directories(... SYSTEM ...)`, as `manta_vendored` and the netCDF
  and Eigen fallbacks in `cmake/MantaDependencies.cmake` do. `-Werror` is on, and
  Eigen's own headers do trip `-Wunused-but-set-variable` under clang — reachable
  only from the pybind11 build, which pulls in `SparseCore`. Adding a dependency
  without `SYSTEM` re-arms that.
* **...but never hand the compiler a directory it already searches.** Passing a
  default system directory with `-isystem` is not a no-op: gcc and clang both
  de-duplicate it, dropping the directory from its proper place at the *end* of the
  system chain and searching it where the flag appeared — ahead of the libstdc++
  headers. `<cstdlib>` then does `#include_next <stdlib.h>`, which only considers
  directories *after* the one holding it, so every translation unit dies with

  ```
  /usr/include/c++/16/cstdlib:83:15: fatal error: stdlib.h: No such file or directory
     83 | #include_next <stdlib.h>
  ```

  `NETCDF_DIR=/usr` did exactly that, which is what a package-manager install means
  on Debian/Ubuntu, and `EIGEN_DIR=/usr/include` did the same. `Makefile.config`
  carried a compiler probe and a `sysinclude` filter for it; **CMake does the
  filtering itself**, stripping anything in `CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES`
  from the flags it generates, so the filter is gone. That is a property to keep
  testing rather than to trust: CI configures a scratch build directory with
  `-DCMAKE_PREFIX_PATH=/usr` and greps `compile_commands.json`, on *every* leg,
  because the implicit-directory list comes from a compiler probe whose output
  differs between gcc and clang.
* **`FindBLAS` left to itself does not pick what `-lblas` picked, and the
  difference can be fatal to the Python module only.** `MANTA_BLAS_VENDOR` defaults
  to `Generic` — a plain `-lblas`, the distribution's alternatives symlink — for
  this reason. Asked freely, `FindBLAS` walks its vendor list and on a box with
  Intel's libraries takes the *layered* MKL link (`mkl_gf_lp64` + `mkl_gnu_thread`
  + `mkl_core` + `libgomp`). Both are MKL on such a box; `-lblas` there is
  `libmkl_rt`, the dispatch layer that initialises itself. The layered link with
  the GNU threading layer is unsafe to `dlopen`, and importing a C extension *is* a
  `dlopen`: the module died mid-solve and took the interpreter with it, leaving
  pytest to report a bare exit 2 with no traceback, while the standalone solver —
  linked from the very same objects — ran the whole regression suite. **That
  asymmetry is the signature**, and the obvious reading of it ("CMake found MKL,
  the Makefile found `-lblas`, so this is reference-versus-vendor") is wrong.
* **A find_package can be overruled by a dependency's own config file, silently.**
  `SUNDIALSConfig.cmake`, when SUNDIALS was built with LAPACK on, writes the BLAS
  *its own* configure resolved straight into the cache as a hardcoded path list,
  before calling `find_dependency(LAPACK)`. `set(... CACHE ...)` without `FORCE` is
  a no-op when the entry exists — but when it does not, CMake *removes the normal
  variable of the same name from the calling scope*, and `FindBLAS` leaves
  `BLAS_LIBRARIES` a normal variable. So MaNTA's choice was replaced by whatever
  machine SUNDIALS happened to be built on, and `BLAS::BLAS` followed it.
  **Ordering alone does not save you** — the BLAS block already ran first.
  `cmake/MantaDependencies.cmake` writes its answer into the cache with `FORCE`,
  and the configure summary prints `BLAS_LIBRARIES` *after* `find_package(SUNDIALS)`
  precisely so a recurrence is visible rather than inferred.
* **`CMAKE_CXX_COMPILER_ID` distinguishes gcc from clang** for the few flags that
  differ: `-fprofile-abs-path` (gcc-only, a hard error on clang),
  `-fno-inline-small-functions` / `-fno-default-inline` (gcc-only, ignored with a
  warning by clang), and `MANTA_GCOV`, which is `gcov-15` for gcc but has to be
  `llvm-cov gcov` for clang. `-flto=auto` is fine on both — clang has accepted it
  as a spelling of full LTO since clang 17, and the CI matrix starts at 19. Two
  flags are *probed* with `check_cxx_compiler_flag` rather than assumed:
  `-march=native` and `-Wno-invalid-feature-combination`, the latter because an
  unknown `-Wno-` name is itself an error under `-Werror`.
* **`MANTA_GCOV` must match the compiler that wrote the `.gcno` files**, and it
  is derived from `CMAKE_CXX_COMPILER_VERSION`, not from the compiler's *name*.
  The Makefile substituted `gcov` for `g++` in `$(CXX)`, which works for `g++-15`
  and does nothing whatever for `/usr/bin/c++` — leaving a bare `gcov`, i.e. the
  distribution's default. That is gcov-13 on ubuntu-24.04, whose image ships gcc
  12/13/14 with 13 as the default while the workflow builds with `g++-15`. gcovr
  then exits 3 with `AdjointVectors.gcno:version 'B42*', prefer 'B33*'`, which it
  promotes to a hard error, and the coverage job fails with exit 64 on CI while
  passing locally. Note how easily this hides: on a box whose default `gcov`
  happens to match its default `c++` — as the development box does — the broken
  derivation gives the right answer. Under the Makefile it additionally needed
  `GCOV` derived *outside* `ifdef COVERAGE`, because the parent make ran gcovr and
  only the child had `COVERAGE=on`; a Coverage build directory has no such split.
* **A `add_custom_target` COMMAND is not shell-quoted unless you say `VERBATIM`,
  and the failure is a shell syntax error attributed to nothing.** The `coverage`
  target passes gcovr a filter regex — `[A-Za-z0-9_]+\.(cpp|hpp)$` — and without
  `VERBATIM` the parentheses reached `/bin/sh` bare:

  ```
  /bin/sh: 1: Syntax error: "(" unexpected
  ```

  after all three suites had run to completion, so the cost was twenty-three
  minutes of instrumented tests and no report. The same target passes
  `MANTA_GCOV` around, which is the two words `llvm-cov gcov` under clang, so it
  has a second reason to need it. Any custom command carrying a regex, a glob or
  an argument with a space wants `VERBATIM`.
* **`gcovr` is looked for beside `Python3_EXECUTABLE`, not just on `PATH`.**
  `pip install gcovr` into a virtualenv puts it in that environment's `bin/`,
  which is on `PATH` only if the environment has been activated — and the point of
  CMake finding the interpreter itself is that it need not have been. Searching
  `PATH` alone made the `venv` and `coverage` targets disagree about whether gcovr
  existed.
* **A restart used to have its trace thrown away.** `setInitialConditions`
  finished every restart with `EvaluateLambda()`, which sets `lambda` to
  `{{u}}` -- the average of the two cell traces (`DGSoln.hpp`) -- and that is not
  the equation `lambda` solves. The HDG trace row is
  `Csigma sigma + Cq q + G_c u + H lambda = L(t)`, so applying the average to a
  file that already holds a converged trace replaces it with something that
  solves nothing: measured on a `TestDiffusion` round trip at
  `Absolute_tolerance = 1e-8`, that one call takes the weighted residual from
  2.6e-3 to 556. It is why a restart needed roughly ten times as many residual
  evaluations inside `IDACalcIC` as a cold start. Note the reordering that went
  with it: `ApplyDirichletBCs` now runs *after* the trace is settled, since
  `EvaluateLambda` overwrites every entry including the boundary ones, so in the
  old order the Dirichlet data was applied and then immediately discarded.

  **The trace is kept whenever the *mesh* matches, not only the discretisation.**
  `lambda` has no polynomial degree — `DGSoln::Map` gives it `nCells + 1` entries
  and no basis — so a change of degree leaves it transferable verbatim even though
  `copy()` refuses the state as a whole. That matters beyond `lambda` itself,
  because the `q` row carries a `<lambda, v n>` term: on a `LinearDiffusion`
  restart coarsened from `k = 4` to `k = 3` at `atol = 1e-10`, keeping the trace
  takes the `q` block from 7.3e7 to 3.2e-7. Only a genuine remesh rebuilds it.
* **`sigma` is loaded on a copy-path restart, not recomputed, and that is a
  measurement too.** `DGSoln::copy` brings `sigma` across with everything else and
  `ApplyDirichletBCs` touches only `lambda`, so `AssignSigma` was rebuilding it
  from bit-identical inputs — at the price of a full `ComputePhysics` over every
  node, which is *exactly one residual evaluation's worth of physics*
  (`residual` makes the same call, `SystemSolver.cpp:1329`). It also evaluates
  `Sources` for every variable and `AuxG` for every auxiliary one and drops both,
  since only the flux is used. On a copy-path restart, which now skips
  `IDACalcIC` entirely, that was half of all the physics `initialize()` did.

  The trade is real and small: a rebuilt `sigma` satisfies its row *exactly*,
  where the file's satisfies it only as well as the previous run's Newton
  converged. Measured on the `sigma` row, recomputed against loaded —
  `LinearDiffusion` 4.3e-19 / 1.5e-18 (both round-off), `AuxVarTest` 6.9e-18 /
  5.6e-9, `nonlin` 3.5e-18 / 4.1e-7. No outcome moves: the whole restart
  round-trip tolerance matrix is unchanged. And the loaded value is the more
  faithful one — it is the `sigma` the previous run was actually integrating with
  when it wrote the file, where the rebuilt one is a state that run never had.
  The *projection* path still rebuilds it, because a degree change leaves the
  stored coefficients in the wrong space.
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
  from the directories listed in `cmake/MantaCleanData.cmake`, sparing
  `*.ref.nc` / `*.ref.dat`.
  `Tests/UnitTests` is deliberately absent: its data files are tracked test
  *inputs* — `testic.nc` (`AutodiffTest.cpp`) and `MatrixDiffusion.restart.nc`
  (`SystemSolverTests.cpp:378`) — and the second has no `.ref.` in its name, so
  the keep-pattern would not save it. Check tracked status, not the filename,
  before adding a directory there. Unit-test output now lands in the **build
  directory**, because that is where CTest launches the binary — the repo root is
  still swept, so a tree carrying output from the Makefile era is tidied rather
  than stranded. `python` is absent for a
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
* **The extension's ABI suffix and the interpreter that imports it now come from
  the same place, and that retired a whole class of failure.** Worth knowing what
  it was, because a report of it may still be in flight. The Makefile named the
  module from `$(PYTHON_CONFIG) --extension-suffix` and took its headers from the
  same program, so the two matched each other but not necessarily the interpreter
  that would import them; `PYTHON_CONFIG` fell back to plain `python3-config`,
  which follows the distribution's unversioned `python3` symlink. On a box whose
  `python3` had moved ahead of `.venv`, `make python` *succeeded* while
  `python_tests`, `stubs-check` and `typecheck` all failed — each with a message
  pointing somewhere else. `stubs-check` was the worst: regenerating the stub
  needs the import too, so it failed to write one and then reported the committed
  `_manta.pyi` stale, which was a claim about a tracked file that was fine.
  `cmake/MantaPython.cmake` finds **one** interpreter — preferring `.venv`, or
  `$VIRTUAL_ENV` — and pybind11 derives the headers and the suffix from it, so the
  three cannot disagree. Name a different one with `-DPython3_EXECUTABLE=...`;
  `setup.py` passes `sys.executable` for the same reason, so `pip install .`
  always builds for the interpreter doing the installing. If an import of `_manta`
  ever does fail, `ls python/manta/*.so` against
  `python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))'`
  is still the check — but the answer is now a build directory configured for a
  different interpreter, not a silent fallback.
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
* ~~Restarting is fragile at tight tolerances, more so with `nAux > 0`.~~ Fixed.
  All three regression round trips now survive `1e-6 / 1e-8`; the ceiling that
  remains at `1e-8 / 1e-10` belongs to the cases (`MatTest`'s *uninterrupted* run
  fails there too) rather than to the restart path. Two fixes closed it, neither
  in the restart machinery: `setInitialConditions` discarding the converged trace
  a restart file carries, and `AuxVarTest`'s missing `dSigma_dPhi` block.
  `MatTest` stays at `1e-4` for cost, not capability — 1e-6 takes 101 s against
  6.0 s.
  A coupled restart is not in that class either — `psi` is copied out of the
  file and, being differential, held fixed by `IDACalcIC`, so it round-trips
  bit for bit at whatever tolerance the run itself survives.
* **No field model is registered anywhere in the tree**, so `FieldModel` has
  nothing to name in the shipped binary and there is no coupled regression case.
  The two models that exist are unregistered fixtures under `Tests/UnitTests`.
  `Tests/README.md` names what this leaves uncovered: nothing exercises the
  coupled path through a `.conf` file, so the config plumbing and the netCDF
  group have unit-test cover only.
* **A field model cannot be reached from Python.** `FieldModel` has no pybind11
  class and `FieldModel` is a `ProblemSelection` key, so it is an error in a
  `Runner.configure` dict. `PyRunner::configure` therefore *refuses* a restart
  file whose `nField` is nonzero, by name, rather than reading `psi` into a
  vector with no field block — which would surface as an `nVars`/`nAux`/
  `nScalars` length complaint naming three things that are all fine.
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
  run by the `python` CTest test (`pytest.ini` is `testpaths = python/Tests`)
  because it needs `desc` and `optimistix`. `Tests/README.md` records this.
* The `UseMMS` options on `LinearDiffusion` and `LinearDiffSourceTest` have been
  removed: the first's manufactured solution did not satisfy its own boundary
  conditions, and the second never applied `MMS_Source` at all. Order of accuracy
  is measured by `Tests/UnitTests/MMSConvergenceTests.cpp`, which builds its own
  manufactured problems.
