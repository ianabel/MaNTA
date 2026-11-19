# Building MaNTA: flags, toolchain and dependencies

What the build does and why, and the traps in compilers, CMake and the dependencies. `CLAUDE.md` has the commands.

## Assertions in a Release build

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
do — see the phase breakdown in `parallelism.md`, where it moves the dominant cost
from a global factorisation to the per-cell loops.

## Where the build lives

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

## Out-of-tree builds

`cmake --install build --prefix ...` installs the headers under
`$PREFIX/include/manta`, `libmanta.so`, and `manta.pc`. **`manta.pc` is written at
*install* time, not configure time**, so `--prefix` is recorded correctly and
`DESTDIR` is honoured; a plain `configure_file` bakes in whatever the prefix was
when cmake last ran, and the first symptom of that is a plugin compiling against
the wrong headers. It also carries the include directories of what the installed
headers include: the vendored toml11, installed under `include/manta/extern`, and
SUNDIALS, Eigen, Boost and netCDF wherever this configure found them (`manta_deps`'s
usage requirements, evaluated at generate time), less the compiler's own search
directories -- without them the documented plugin build did not compile anywhere
SUNDIALS was not in a system directory. The installed binaries likewise carry an
RPATH to SUNDIALS's library directory, named from the linked targets rather than
left to `INSTALL_RPATH_USE_LINK_PATH`, which skips anything inside the project --
and `./build_sundials` installs into `./sundials/install`, so the documented setup
installed a solver that could not find `libsundials_ida` at startup. There is
deliberately **no installed CMake package**: an
exported target would invite `target_link_libraries(mycase manta)`, and linking
`libmanta` is exactly the mistake below. A physics case built as a shared object and named
in the config's `PhysicsPlugins` array is dlopened by `runManta` before
`InstantiateProblem`; its static initialiser registers it into the same
process-global map. CI's `Out of tree` job runs both routes the way a user does --
this install, a plugin built with pkg-config's flags, and `pip install .` into a
fresh venv driving a Python case through `manta` -- from `Tests/OutOfTree/`,
whose files `docs/out_of_tree.rst` includes. Two traps, neither of which is a link error:

* **A plugin must be compiled with the flags `pkg-config --cflags manta`
  reports.** Eigen aligns its types to the widest vector unit the compiler knows
  about and inlines its expression templates into both sides of the boundary, so
  a plugin built without the core's `-march=` faults inside an aligned AVX-512
  load (`_mm512_load_pd`) the first time the solver touches its state. `manta.pc`
  records the *concrete* architecture (`-march=znver4`, not `native`) so a
  mismatch is a compile error rather than a run-time crash. `-DEIGEN_USE_BLAS`
  and `-DEIGEN_DONT_PARALLELIZE` travel for the same reason.
* **A plugin must not link `-lmanta`.** The solver links the core objects
  directly, so a plugin that also pulled in `libmanta.so` would get a second copy
  of `PhysicsCases::map` and register into a map the solver never reads —
  silently. Compile against the headers alone and let the loader bind the MaNTA
  symbols to the host, which is linked `-rdynamic` for exactly that.
  `libmanta.so` is for embedding the solver in another program.

## Toolchain and dependency traps

**`extern/autodiff` points at a fork, and `git submodule update --remote` would
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

**Eigen 3.4.x and 5.0.x are both supported, and `EIGEN_VERSION_AT_LEAST` cannot
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

**g++-14.2.0 miscompiled this tree at `-O3 -flto -march=native`, and the
symptom is a wrong number rather than a crash.** Measured 2026-08-23 on
g++-14.2.0, znver2; the README's floor for gcc is 14.3 because of it.
**g++-14.3.0 does not reproduce it** (2026-11-06): all three suites pass at the
release flags, and with the inert member below in place the reproduction test
and three full unit-suite runs are clean. Whether 14.3 fixed it or the rebuilt
test misses it is not known, which is why two CI legs carry gcc 14 again --
the ubuntu-24.04 image's, which is 14.3.0 there too, and RHEL 10's system
compiler, on `almalinux:10`. Nothing in CI builds 14.2.0. RHEL 10 keeps
gcc 14 for the life of the release, so this is a platform question rather than
a curiosity.

The trigger is **any change to `SystemSolver`'s member layout**. Adding one
inert member — a `bool` and an unused `std::vector<double>`, referenced by no
code anywhere — took a clean tree from 12/12 passing to 8/12 failing, on a test
that densely finite-differenced `residual()` inside the test translation unit.
Only the `AuxDiffusion` cases failed, plain and superconvergent, at every `k`,
with a relative drift of about 1.24: an O(1) error, not a tolerance one.
Nothing else in the suite ever failed.

**The test of that shape is `an_aux_jacobian_matches_a_finite_difference_of_its_residual`**
(`SolveJacTests.cpp`): the same `AuxDiffusion` case, a dense finite difference
of `residual()` inlined into the test TU, checked as `J dy = g` at k = 1-3,
plain and superconvergent. With an inert member added to `SystemSolver` it is
the reproduction. The defect was never diagnosed, only bounded, so treat it as
live on 14.2.

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

**g++-14 was dropped from CI for this and is back on two legs**, `Build + tests
(g++-14)` (the runner image's, 14.3.0) and `Build + tests (almalinux:10, gcc 14)`. The
coverage job stays on g++-15, and the Eigen 5.0.1 legs on g++-15 and
clang++-19. A failure on either gcc 14 leg in a test that differences the
residual, by the drift above, is this defect rather than the change under
review. Both pass every suite (2026-11-07): the runner's g++ 14.3.0 and
AlmaLinux 10's gcc 14.3.1, Red Hat's build of the compiler RHEL 10 keeps for
the life of the release. The release build warns below 14.3; `TODO` has the
full reproduction.

**gcc and clang do not diagnose the same things, so build with clang
occasionally** — that is what CI's clang matrix legs are for. (CI is nine
`build-and-test` legs: g++-14/15/16 and clang++-19/20/21 against the distro's
Eigen 3.4.0, plus g++-15 and clang++-19 against Eigen 5.0.1, plus g++-15 with
`MANTA_OPENMP=ON`; then three container legs, Fedora with and without OpenMP
and AlmaLinux 10 on its system gcc 14, which keep the build's notions of a
system prefix — `/usr/lib64`, pkg-config-discovered netCDF, EPEL — from
quietly becoming Ubuntu-specific.) gcc never
diagnoses a polymorphic base with a non-virtual destructor; clang does
(`-Wdelete-non-abstract-non-virtual-dtor`), and it reports it at the point of
*destruction* inside libstdc++, once per instantiating translation unit, which
makes the message look like a standard-library problem rather than yours. That is
how `MagneticField`'s missing virtual destructor was found.

There is no global `-Wno-` in the build: third-party headers are silenced by
`-isystem` at source. Prefer that to a blanket suppression on the project, which
outlives whatever it was added for and goes on hiding defects in *our* code --
a global `-Wno-parentheses` once took gcc's `-Wdangling-else` with it.

**With clang, the libstdc++ version is part of the configuration.** clang selects
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

**And a second: `std::format` of a `std::vector` needs libstdc++ 15.** P2286's
range formatters landed there, so formatting a container whole compiles on a box
with g++-15 and is a `static_assert` inside `<format>` on CI's clang legs —
*"std::formatter must be specialized for each type being formatted"*, naming
`<format>` and `variant` and nothing in this tree. What triggered it was adding
`std::vector<unsigned int>` to `ConfigSchema::Value` for the ladders without
adding it to the vector branch of `main.cpp`'s `defaultText`, so it fell through
to the generic `std::format("{}", x)`. Every vector alternative of that variant
has to be named in that branch, which formats element by element. All five gcc
legs were green and all five clang legs red, which is the signature — a build
error that splits the matrix by compiler is almost always the standard library
rather than the compiler.

**Reproduce it locally before pushing a fix rather than after.** Both halves of
CI's configuration are installable here, and the check costs a syntax-only pass:

```sh
clang++-19 -std=c++23 --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/14 \
    -fsyntax-only -I. -Iextern/toml11/include main.cpp
```

That gave 4 errors before the fix and 0 after — the same 4 CI reported. Run it
*without* the fix too: a clean result from a probe that never reproduced the
failure is evidence of nothing.

**Third-party includes must be `SYSTEM`.** An imported target's include
directories are already treated that way; anything added by hand needs
`target_include_directories(... SYSTEM ...)`, as `manta_vendored` and the netCDF
and Eigen fallbacks in `cmake/MantaDependencies.cmake` do. `-Werror` is on, and
Eigen's own headers do trip `-Wunused-but-set-variable` under clang — reachable
only from the pybind11 build, which pulls in `SparseCore`. Adding a dependency
without `SYSTEM` re-arms that.

**...but never hand the compiler a directory it already searches.** Passing a
default system directory with `-isystem` is not a no-op: gcc and clang both
de-duplicate it, dropping the directory from its proper place at the *end* of the
system chain and searching it where the flag appeared — ahead of the libstdc++
headers. `<cstdlib>` then does `#include_next <stdlib.h>`, which only considers
directories *after* the one holding it, so every translation unit dies with

```
/usr/include/c++/16/cstdlib:83:15: fatal error: stdlib.h: No such file or directory
   83 | #include_next <stdlib.h>
```

`NETCDF_DIR=/usr` or `EIGEN_DIR=/usr/include` -- what a package-manager install
means on Debian/Ubuntu -- would do exactly that. **CMake does the filtering
itself**, stripping anything in `CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES` from the
flags it generates. That is a property to keep
testing rather than to trust: CI configures a scratch build directory with
`-DCMAKE_PREFIX_PATH=/usr` and greps `compile_commands.json`, on *every* leg,
because the implicit-directory list comes from a compiler probe whose output
differs between gcc and clang.

**`FindBLAS` left to itself does not pick what `-lblas` picked, and the
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
asymmetry is the signature**, and the obvious reading of it ("reference
BLAS against vendor BLAS") is wrong.

**A find_package can be overruled by a dependency's own config file, silently.**
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

**`CMAKE_CXX_COMPILER_ID` distinguishes gcc from clang** for the few flags that
differ: `-fprofile-abs-path` (gcc-only, a hard error on clang),
`-fno-inline-small-functions` / `-fno-default-inline` (gcc-only, ignored with a
warning by clang), and `MANTA_GCOV`, which is `gcov-15` for gcc but has to be
`llvm-cov gcov` for clang. `-flto=auto` is fine on both — clang has accepted it
as a spelling of full LTO since clang 17, and the CI matrix starts at 19. Two
flags are *probed* with `check_cxx_compiler_flag` rather than assumed:
`-march=native` and `-Wno-invalid-feature-combination`, the latter because an
unknown `-Wno-` name is itself an error under `-Werror`.

**`MANTA_GCOV` must match the compiler that wrote the `.gcno` files**, and it
is derived from `CMAKE_CXX_COMPILER_VERSION`, not from the compiler's *name*.
Deriving it from the name turns `/usr/bin/c++` into a bare `gcov`, the
distribution's default: gcov-13 on ubuntu-24.04, whose workflow builds with
`g++-15`. gcovr then exits 3 with `AdjointVectors.gcno:version 'B42*', prefer
'B33*'` and the coverage job fails on CI while passing on a box whose default
`gcov` happens to match its default `c++`.

**A `add_custom_target` COMMAND is not shell-quoted unless you say `VERBATIM`,
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

**`gcovr` is looked for beside `Python3_EXECUTABLE`, not just on `PATH`.**
`pip install gcovr` into a virtualenv puts it in that environment's `bin/`,
which is on `PATH` only if the environment has been activated — and the point of
CMake finding the interpreter itself is that it need not have been. Searching
`PATH` alone made the `venv` and `coverage` targets disagree about whether gcovr
existed.

**gcov counts a templated line once per instantiation**, which makes
`NetCDFIO.hpp` and `util/trapezoid.hpp` look far worse than they are. Judge
those by distinct uncovered lines; `Tests/README.md` has the numbers.
