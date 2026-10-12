# A user convergence test for KINSOL (upstream SUNDIALS change)

Date: 2026-10-12
Status: design; nothing written against SUNDIALS. Branch `gp-toy-model`.
Companion to `2026-10-11-steady-state-termination-design.md`.

## Why

MaNTA's `SteadyStateTest = "NewtonStep"` stops a steady solve on IDA's
corrector test -- `ss * ||delta||_WRMS <= epsNewt`, `delta` the Newton
correction -- and KINSOL cannot express that. So `NewtonStep` runs on SUNDIALS'
generic Newton (`SUNNonlinSol_Newton`), which takes a convergence test as a
callback because IDA and CVODE plug theirs in that way. That works, but it
splits MaNTA's steady solver in two: KINSOL for `Residual`, `SUNNonlinSol` for
`NewtonStep`, with the Jacobian-reuse policy reimplemented in a linear-solve
callback for the second.

This is the KINSOL change that would let `NewtonStep` go back to KINSOL: a
user-supplied convergence test, called on the Newton step before KINSOL
evaluates the system function at the new iterate. It is small, it follows a
precedent KINSOL already has (`KINSetDampingFn`, `KINSetDepthFn`), and it is
useful well beyond MaNTA: any caller with a stopping criterion of its own --
per-component tolerances, a masked norm, a noise-aware floor -- has nowhere to
put it today.

## What KINSOL does now

Read at `LLNL/sundials` `develop` 4f3f8063f (2026-10-08), the same as v7.9.0 in
everything below.

* **The Newton loop** (`kinsol.c:618-710`). Under `KIN_NONE`: `KINLinSolDrv`
  solves for the step `pp = J^-1 (-F(uu))`; `KINFullNewton` (1301) scales it to
  `mxnewtstep` and the constraints, forms `unew = uu + pp` and evaluates
  `F(unew)`; `KINStop` (1821) then tests. `KIN_LINESEARCH` differs only in the
  middle step (`KINLineSearch`, 1448).
* **The tests** (`KINStop`): `max_i |fscale_i F_i(unew)| <= fnormtol`, then the
  scaled step `max_i |delta_i| / (1/uscale_i + |unew_i|) <= scsteptol`, which
  returns `KIN_STEP_LT_STPTOL` -- "stuck", a warning, not convergence. Then max
  iterations, five consecutive maximum-length steps, and residual monitoring
  for modified Newton.
* **The initial guess** (`KINSolInit`, 1186): `fmax <= 0.01 fnormtol` returns
  `KIN_INITIAL_GUESS_OK` without iterating.
* **Fixed point and Picard** (`KINPicardAA` 2222, `KINFP` 2453) test
  `fnorm <= fnormtol` on their own residual at 2362 and 2581.

There is no hook between the step and the evaluation at the new iterate, and
no way to replace the tests. A search of all 115 remote branches (fetched
2026-10-12) for a convergence or stopping hook in `src/kinsol` or
`include/kinsol` finds none. The KINSOL work in flight is unrelated:
`feature/sparse-fdjac` (a coloured finite-difference Jacobian for sparse
matrices), `feature/temp-vector-stack`, and the October deprecation removals
(`KINGetWorkSpace` and friends, none of which MaNTA calls).

## The change

### API (`include/kinsol/kinsol.h`)

```c
/* status a convergence test reports through its last argument */
#define KIN_CONVTEST_CONTINUE  0
#define KIN_CONVTEST_CONVERGED 1
#define KIN_CONVTEST_DIVERGED  2

/* return flags */
#define KIN_CONVTEST_FN_ERR   -20  /* the test returned nonzero */
#define KIN_CONVTEST_NONCONV  -21  /* the test reported divergence */

typedef int (*KINConvTestFn)(long int iter, N_Vector u_val, N_Vector delta,
                             N_Vector f_val, void* user_data, int* status);

SUNDIALS_EXPORT int KINSetConvTestFn(void* kinmem, KINConvTestFn ctest_fn);
```

The shape copies `KINDampingFn` (`kinsol.h:87`): the iteration number first, the
vectors, then `user_data` -- the pointer from `KINSetUserData` -- and the
result through a pointer, with the return value reserved for failure of the
function itself. `-18` and `-19` are the damping and depth hooks' errors, so
`-20` and `-21` are the next free codes; `KINGetReturnFlagName` gains both.

* `iter` is `kin_nni`, the current iteration, from 1. A test with state across
  iterations -- IDA's rate estimate -- resets on `iter == 1`.
* `u_val` is the current iterate `uu`; `delta` the step KINSOL is about to take,
  *after* the `mxnewtstep` and constraint scaling, so the test sees the step
  that will actually be applied; `f_val` is `F(uu)`, which the step was
  computed from. Read-only, all three.
* Passing `NULL` removes the hook and restores KINSOL's own tests.

### Semantics

With a hook set:

* **`KIN_NONE` and `KIN_LINESEARCH`**: the hook is called after the step is
  final and before the system function is evaluated at `uu + delta`.
  * `CONVERGED`: `uu <- uu + delta`, no further evaluation, `KINSol` returns
    `KIN_SUCCESS`. Under `KIN_LINESEARCH` the full step is taken without a line
    search, on the grounds that the test has just called it small enough to
    stop on. This is IDA's semantics, and it saves one system evaluation per
    solve.
  * `CONTINUE`: KINSOL proceeds exactly as now -- `KINFullNewton` or
    `KINLineSearch` evaluates at the new iterate, and the loop goes on.
  * `DIVERGED`: `KINSol` returns `KIN_CONVTEST_NONCONV` with `uu` at the last
    iterate, the step not taken. The caller decides what that means (MaNTA damps
    harder).
* **`KIN_PICARD` and `KIN_FP`**: called once the new iterate is formed, with
  `delta = unew - uu`, in place of the `fnorm <= fnormtol` test. `CONVERGED`
  returns `unew`; there is no evaluation to save, since forming `unew` needed
  `G(uu)`.
* **KINSOL's own convergence tests are replaced, not supplemented**: no
  `fnormtol` test in `KINStop` or `KINSolInit`, and no scaled-step test. A
  caller who wants the residual test as well reads `f_val` in its hook.
  Everything that is not a convergence test stays: `mxiter`
  (`KIN_MAXITER_REACHED`), `KIN_MXNEWT_5X_EXCEEDED`, system-function failure
  and recovery, the constraint handling, and residual monitoring for modified
  Newton, which uses `fnorm` only to decide when to refresh the Jacobian.
* **The initial guess is not tested on its own**, because there is no step to
  test until one has been solved for. The first hook call, at `iter == 1`, is
  the equivalent, and a test that passes there costs one evaluation, one setup
  and one solve, against `KIN_INITIAL_GUESS_OK`'s one evaluation.
* **Statistics.** `nfe` does not count the skipped evaluation. `KINGetFuncNorm`
  reports the norm of the last residual evaluated, which after a converged
  hook is `F` at the iterate *before* the accepted step; the docs must say so.
  `KINGetStepLength` reports the accepted step.

### Implementation sketch (`src/kinsol`)

About 120 lines of C, in four places:

* `kinsol_impl.h`: `KINConvTestFn kin_ctest_fn;` and an internal flag
  `kin_ctest_converged`; initialised in `KINCreate`.
* `kinsol_io.c`: `KINSetConvTestFn`, beside `KINSetDampingFn`, and the two flag
  names in `KINGetReturnFlagName`.
* `kinsol.c`, `KINFullNewton` and `KINLineSearch`: after the constraint block
  and before the evaluation loop, call the hook. On `CONVERGED` form `unew`, set
  `kin_ctest_converged` and return `KIN_SUCCESS` without evaluating; on
  `DIVERGED` return a new internal `CTEST_DIVERGED`; on a nonzero return,
  `KINProcessError(..., KIN_CONVTEST_FN_ERR, ...)` and return it.
* `kinsol.c`, the `KINSol` loop: after the step, if `kin_ctest_converged`,
  copy `unew` into `uu`, log, and break with `KIN_SUCCESS` without calling
  `KINStop`; map `CTEST_DIVERGED` to `KIN_CONVTEST_NONCONV` without updating `uu`.
  In `KINStop`, skip the `fnormtol` and `scsteptol` blocks when a hook is set;
  in `KINSolInit`, skip the initial-guess test. In `KINPicardAA` and `KINFP`,
  replace the `fnorm` tests at 2362 and 2581 by the hook when it is set.

### What else a pull request needs

From `CONTRIBUTING.md`: "user guide additions or changes as appropriate, has been
tested, and includes a test for any added features", BSD-3 licensing, the
author/date/licence header on any new file, and the DCO, affirmed by opening
the PR. Concretely:

* **Tests.** `test/unit_tests/kinsol/C_serial/kin_test_convtest.c` (and its
  `CMakeLists.txt` entry): an IDA-style test on a small nonlinear system under
  `KIN_NONE` and `KIN_LINESEARCH`, checking convergence and that `nfe` equals
  `nni` -- the converged iterate not evaluated; a hook reporting `DIVERGED`,
  checking `KIN_CONVTEST_NONCONV` and an unchanged `uu`; a hook returning
  nonzero, checking `KIN_CONVTEST_FN_ERR`; and the same under `KIN_FP`.
  `test/unit_tests/kinsol/gtest/test_kinsol_error_handling.cpp` for the two
  codes. Every existing test and example must be unchanged with no hook set.
* **Docs.** `doc/kinsol/guide/source/Usage/index.rst`: the optional-input table,
  the function, the `KINConvTestFn` typedef with the user-supplied functions,
  and the two return flags; `doc/kinsol/guide/source/Mathematics.rst`: the
  stopping criteria; `CHANGELOG.md` and `doc/shared/RecentChanges.rst`.
* **Fortran.** `swig/kinsol/fkinsol_mod.i`, regenerating
  `src/kinsol/fmod_int32` and `fmod_int64`.
* **Python.** `bindings/sundials4py/kinsol/kinsol_usersupplied.hpp`: a
  `convtestfn` slot in `kinsol_user_supplied_fn_table`, a wrapper returning
  `std::tuple<int, int>` as `kinsol_dampingfn_wrapper` returns its factor, and
  a `BIND_KINSOL_CALLBACK(KINSetConvTestFn, ...)` in `kinsol.cpp`; a test in
  `bindings/sundials4py/test/test_kinsol.py`. The repository's own
  `.agents/skills/sundials-devs-python-callbacks` describes the pattern.

`CONTRIBUTING.md` also says the team "does not have the resources to review and
take in large additions"; this is a small one, but an issue first, describing
the use case and the API above, costs little and finds out.

### Alternative: a built-in step test

KINSOL could instead offer IDA's test itself -- say
`KINSetStepTest(kinmem, epsNewt)`, WRMS with `u_scale` as the weights and IDA's
rate factor. Less API, and arguably more in KINSOL's style. It would not carry
MaNTA: `NewtonStep` masks the algebraic components under
`SuppressAlgebraicError`, takes the larger of IDA's rate and the last ratio, and
applies IDA's divergence test only to its undamped probes -- each measured
necessary (see the termination design's "As built"). A hook covers all of that
and any test anyone else needs; a built-in covers one.

## MaNTA, once KINSOL has it

The hook would land in some release after 7.9 and MaNTA's floor is 7.1.0, so
the `SUNNonlinSol` path stays until the floor passes that release. Until then:

* `cmake/MantaDependencies.cmake`: `check_symbol_exists(KINSetConvTestFn
  kinsol/kinsol.h MANTA_HAVE_KINSOL_CONVTEST)` after `find_package(SUNDIALS)`.
* With it, `steadyNewtonSolve` becomes a `KINSol` call under `KIN_NONE`:
  * `nlsConvTest` becomes the hook almost as it is -- the rate state resets on
    `iter == 1`, `DIVERGED` replaces `SUN_NLS_CONV_RECVR`, and the masked WRMS
    norm and `nlsStrictDivergence` carry over unchanged.
  * `KINSetMaxSetupCalls(newtonJacReuse)` replaces the rebuild in `nlsLSolve`,
    which exists only because `SUNNonlinSol_Newton` sets up once per solve.
  * The steady-residual norm at the start of a step, for SER, is recorded in
    the first `SteadyResidual` call of each `KINSol`, as `nlsSys` records it at
    `ycor = 0` now.
  * `KIN_CONVTEST_NONCONV` and `KIN_MAXITER_REACHED` both mean "not converged",
    which the loop already answers by damping.
* The sweep budget is unchanged: `KINSolInit` evaluates `F(u0)` once, and each
  iteration then builds, solves, tests and evaluates only if not converged --
  `1 + n` flux sweeps and `n` derivative sweeps over `n` iterations, as now.
  `a_newton_step_solve_spends_one_flux_and_one_derivative_sweep_per_iteration`
  pins it either way, and every `NewtonStep` test should pass unchanged on both
  paths; a CI leg on a SUNDIALS with the hook would run them on the KINSOL one.

## Open questions

* **Replace or supplement.** Replacing KINSOL's tests is IDA's model and the
  simplest to document; supplementing (converge when either passes) keeps
  `fnormtol` meaningful. Replacing is proposed, since a hook can read `f_val`.
* **The initial guess.** Skipping `KIN_INITIAL_GUESS_OK` under a hook costs an
  already-converged caller one setup and one solve. An optional
  `KINSetConvTestInitialGuess` -- call the hook with `delta = NULL` once, before
  iterating -- would avoid it, at more API.
* **`f_val` under `KIN_FP`**, where KINSOL holds `G(u)` and the residual is
  `G(u) - u`: pass the residual, which needs one vector operation, or `G(u)`
  documented as such.
