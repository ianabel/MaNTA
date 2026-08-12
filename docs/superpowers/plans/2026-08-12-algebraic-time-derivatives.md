# Algebraic Time Derivatives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute `q'`, `sigma'` and `phi'` at `t0` by differentiating the algebraic constraints, so `SystemSolver::dGdt` — and therefore the dG/dt early-exit gate — sees all four terms of its chain rule instead of only the `u` one.

**Architecture:** A new `computeAlgebraicTimeDerivatives()` assembles the per-cell blocks at `alpha = 0`, substitutes an identity for the `u` row, builds a right-hand side from the known `u'` and central-differenced explicit `d/dt` terms, and solves through the existing `solveJacEq`. The result lands in new `dydtComplete` storage, never in IDA's `dYdt`.

**Tech Stack:** C++23, SUNDIALS IDA, Eigen, Boost.Test.

Spec: `docs/superpowers/specs/2026-08-12-algebraic-time-derivatives-design.md`.

## Global Constraints

- **Never write into IDA's `dYdt`.** It is the state IDA takes its first step from; changing its algebraic entries after `IDA_YA_YDP_INIT` would alter the integration and surface later as a step-size or convergence problem. All results go to `dydtComplete`.
- **`updateBoundaryConditions(t)` must be restored to the original `t`** after every difference. It writes `RF_cellwise` and `L_global` in place and those are what the forward residual reads — CLAUDE.md already records this trap.
- **The regression suite must be bit-identical.** Nothing in the forward path changes and no config arms the gate, so any movement is a defect in this change — most likely the factorisation risk below.
- **`computeAlgebraicTimeDerivatives()` runs only when `CheckObjectiveDecrease` is set.** A run with the gate disarmed pays nothing.
- Central-difference step is `h = sqrt(eps) * max(1.0, |t|)`.
- The gate stays between `initialize()` and `integrate()`. This plan does not move it.
- Build and test with the venv on `PATH`: `export PATH="$PWD/.venv/bin:$PATH"`.

---

## Reference material the implementer must read first

This plan deliberately does **not** transcribe the per-cell block indices. They
are spelled out in three places already and getting one wrong is the single most
likely way to break this silently, so mirror them rather than retyping from
here:

| What | Where |
|---|---|
| The algebraic residual rows being differentiated | `SystemSolver.cpp:1169-1186` (sigma, q, u, Aux) and `:1122` (lambda) |
| Block assembly and `MX` layout, including how `alpha` scales `X` | `SystemSolver.cpp:604-720` (`updateMatricesForJacSolve`) and `Matrices.cpp` |
| Static condensation and back-substitution, i.e. what shape the solve expects | `SystemSolver::solveHDGJac`, `SystemSolver::solveJacEq` |
| The same blocks transposed, which must stay in step | `initializeMatricesForAdjointSolve` |

The DOF order within a cell is `[sigma | q | u | aux]`, then all of `lambda`,
then `mu`. CLAUDE.md calls a wrong column index here "the most common way to
break the solver silently".

---

## File Structure

| Path | Responsibility |
|---|---|
| `SystemSolver.hpp` | `dydtComplete` storage + view; `computeAlgebraicTimeDerivatives()` declaration |
| `AlgebraicDerivatives.cpp` *(new)* | The assembly, the RHS, the solve, the central differences. Kept out of `SystemSolver.cpp`, which is already 84k. |
| `Solver.cpp` | Call it at the end of `initialize()`; point `objectiveIsDecreasing()` at `dydtComplete` |
| `Tests/UnitTests/AlgebraicDerivativeTests.cpp` *(new)* | The five checks in the spec |
| `Makefile:12`, `Tests/UnitTests/Makefile:6,23` | New sources |
| `TODO`, `docs/adjoints.rst`, `CLAUDE.md` | Documentation |

---

### Task 1: Storage that is not IDA's

**Files:** Modify `SystemSolver.hpp`, `Solver.cpp`; Test `Tests/UnitTests/AlgebraicDerivativeTests.cpp` (new), `Tests/UnitTests/Makefile`

**Interfaces:**
- Produces: `SystemSolver::dydtComplete` (a `DGSoln`, mapped over `dydtCompleteMem`), seeded with IDA's `dYdt` at the end of `initialize()`. Tasks 2-3 write its algebraic blocks and read the whole thing.

- [ ] **Step 1: Write the failing test**

Create `Tests/UnitTests/AlgebraicDerivativeTests.cpp`, and add it to
`TEST_SOURCES` (`Tests/UnitTests/Makefile:6`):

```cpp
// The algebraic time derivatives: q', sigma' and phi' at t0, obtained by
// differentiating the constraints that define them.
//
// At t0 IDA leaves those blocks of dydt identically zero -- IDA_YA_YDP_INIT
// computes algebraic values and differential derivatives -- which
// at_t0_only_the_differential_part_of_dydt_exists in SolverLifecycleTests.cpp
// pins. This file is about the vector that fills them in.

#include <boost/test/unit_test.hpp>

#include "CapturedOutput.hpp"
#include "SystemSolver.hpp"
#include "TestDiffusion.hpp"
#include "Types.hpp"

#include <toml.hpp>

using namespace toml::literals::toml_literals;

namespace
{
const toml::value alg_config = u8R"(
    [DiffusionProblem]
    Kappa = 1.0
    Centre = 0.0
)"_toml;

constexpr Index k = 2, nCells = 4;
} // namespace

BOOST_AUTO_TEST_SUITE(algebraic_derivative_tests)

BOOST_AUTO_TEST_CASE(dydtComplete_starts_as_a_copy_of_idas_derivative)
{
    // Separate storage, seeded from IDA's. The separation is the point: writing
    // the algebraic blocks into IDA's own dYdt would change the state it takes
    // its first step from.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile("algderiv_storage");
    sys.setOutputCadence(0.05);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    // The u block is IDA's, and is not zero -- u is differential.
    Value uNorm = 0.0;
    for (Index i = 0; i < nCells; ++i)
        uNorm += sys.dydtComplete.u(0).getCoeff(i).second.norm();
    BOOST_TEST(uNorm > 1e-8, "dydtComplete's u block is empty, so it was never seeded");

    // And it is a distinct object from the one IDA owns.
    BOOST_TEST(sys.dydtComplete.getCoeffMemPtr() != N_VGetArrayPointer(sys.dYdt),
               "dydtComplete aliases IDA's dYdt, so writing to it would change the run");

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_SUITE_END()
```

`getCoeffMemPtr()` is whatever `DGSoln` exposes for the base pointer — check
`DGSoln.hpp` and use the real accessor; if there is none, compare
`&sys.dydtComplete.u(0).getCoeff(0).second(0)` against the corresponding entry
of `N_VGetArrayPointer(sys.dYdt)`.

- [ ] **Step 2: Run it and watch it fail**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make test -j$(nproc) 2>&1 | tail -20
```

Expected: a compile error — `SystemSolver` has no member `dydtComplete`.

- [ ] **Step 3: Add the storage**

In `SystemSolver.hpp`, beside the existing `yJac`/`dydtJac` declarations and
their owning memory (`yJacMem`), add the same pattern:

```cpp
        // The time derivative with its algebraic blocks filled in.
        //
        // IDA's dYdt has zeros in q, sigma and phi at t0 -- IDA_YA_YDP_INIT
        // computes algebraic *values* and differential *derivatives*, so there
        // is no y' for them to fetch. computeAlgebraicTimeDerivatives() solves
        // the differentiated constraints for them and writes the answer here.
        //
        // Here rather than into dYdt because dYdt is the state IDA takes its
        // first step from: changing its algebraic entries after IDACalcIC would
        // alter the integration, and the symptom would be a step-size failure
        // somewhere later rather than anything pointing back here.
        std::vector<double> dydtCompleteMem;
        DGSoln dydtComplete;
```

Match `yJacMem`/`dydtJac`'s construction exactly — same constructor arguments,
same `Map` call, same place in the member initialiser list.

- [ ] **Step 4: Seed it in `initialize()`**

At the end of `initialize()`, after the existing `setJacEvalY(Y, dYdt)`:

```cpp
	// Seed the complete derivative from IDA's. Its algebraic blocks are zero at
	// this point; computeAlgebraicTimeDerivatives() fills them when the gate is
	// armed, and nothing else reads them.
	dydtComplete.Map(dydtCompleteMem.data());
	std::copy_n(N_VGetArrayPointer(dYdt), dydtCompleteMem.size(), dydtCompleteMem.data());
```

- [ ] **Step 5: Run the tests**

```sh
make test -j$(nproc) 2>&1 | tail -6
```

Expected: the new case passes; every existing case still passes. `make
regression_tests` should also be untouched, but the meaningful check for that is
after Task 2.

- [ ] **Step 6: Commit**

```bash
git add SystemSolver.hpp Solver.cpp Tests/UnitTests/AlgebraicDerivativeTests.cpp Tests/UnitTests/Makefile
git commit -m "Give the solver a derivative vector that is not IDA's"
```

---

### Task 2: The solve

**Files:** Create `AlgebraicDerivatives.cpp`; Modify `SystemSolver.hpp`, `Makefile:12`, `Tests/UnitTests/Makefile:23`; Test `Tests/UnitTests/AlgebraicDerivativeTests.cpp`

**Interfaces:**
- Consumes: `dydtComplete` from Task 1; `updateMatricesForJacSolve`, `solveJacEq`, `updateBoundaryConditions`, `setAlpha` from `SystemSolver`.
- Produces: `void SystemSolver::computeAlgebraicTimeDerivatives()`. Task 3 calls it.

- [ ] **Step 1: Write the two failing tests**

Append to `AlgebraicDerivativeTests.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(the_u_block_round_trips_through_the_identity_row)
{
    // The u row of MX is overwritten with the identity and the known u' put in
    // that row of the RHS, so the solve must hand u' back unchanged. It is a
    // free self-check on the substitution and the condensation: if this fails,
    // the q/sigma/phi blocks beside it are meaningless.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile("algderiv_roundtrip");
    sys.setOutputCadence(0.05);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);

    std::vector<Vector> before;
    {
        CapturedOutput quiet;
        sys.initialize();
        for (Index i = 0; i < nCells; ++i)
            before.push_back(sys.dydtComplete.u(0).getCoeff(i).second);
        sys.computeAlgebraicTimeDerivatives();
    }

    for (Index i = 0; i < nCells; ++i)
    {
        const Vector after = sys.dydtComplete.u(0).getCoeff(i).second;
        for (Index j = 0; j < after.size(); ++j)
            BOOST_TEST(after(j) == before[i](j), boost::test_tools::tolerance(1e-12));
    }

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}

BOOST_AUTO_TEST_CASE(the_algebraic_blocks_stop_being_zero)
{
    // The change, stated as a measurement. Before this call q' and sigma' are
    // identically zero, which is what makes dGdt at t0 see only its u term.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    SystemSolver sys(grid, k, &problem);
    sys.setTau(1.0);
    sys.resetCoeffs();
    sys.setInputFile("algderiv_nonzero");
    sys.setOutputCadence(0.05);
    sys.setNOutput(11);
    sys.setInitialTime(0.0);
    sys.setMinStepSize(1e-12);
    sys.setTolerances({1e-8}, 1e-6);

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    Value qBefore = 0.0;
    for (Index i = 0; i < nCells; ++i)
        qBefore += sys.dydtComplete.q(0).getCoeff(i).second.norm();
    BOOST_TEST(qBefore == 0.0, boost::test_tools::tolerance(0.0));

    {
        CapturedOutput quiet;
        sys.computeAlgebraicTimeDerivatives();
    }

    Value qAfter = 0.0;
    for (Index i = 0; i < nCells; ++i)
        qAfter += sys.dydtComplete.q(0).getCoeff(i).second.norm();
    BOOST_TEST(qAfter > 1e-8,
               "q' is still zero after solving for it, so the RHS or the solve is empty");

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}
```

- [ ] **Step 2: Run and watch them fail**

Expected: compile error, `computeAlgebraicTimeDerivatives` not a member.

- [ ] **Step 3: Write `AlgebraicDerivatives.cpp`**

Add `AlgebraicDerivatives.cpp` to `SOURCES` (`Makefile:12`) and
`../../AlgebraicDerivatives.o` to `REQUIRED_OBJECTS`
(`Tests/UnitTests/Makefile:23`). The function has four parts, in order:

1. **Central-difference the explicit `d/dt` terms.** For the boundary data:

```cpp
	const double h = std::sqrt(std::numeric_limits<double>::epsilon()) *
	                 std::max(1.0, std::abs(t0));

	updateBoundaryConditions(t0 + h);
	const auto RF_plus = RF_cellwise;
	const auto L_plus = L_global;

	updateBoundaryConditions(t0 - h);
	const auto RF_minus = RF_cellwise;
	const auto L_minus = L_global;

	// Restore. These two arrays are what the forward residual reads, and
	// leaving them at t0 - h would corrupt the run -- see CLAUDE.md on
	// RF_cellwise and L_global.
	updateBoundaryConditions(t0);
```

with `dRF_dt = (RF_plus - RF_minus) / (2h)` and likewise for `L`. Do the same
for `sigmaHat` and `AuxG` by evaluating the physics hooks at `t0 ± h` with the
state held at `yJac`.

2. **Assemble at `alpha = 0`.** Save `alpha`, `setAlpha(0.0)`, call
   `updateMatricesForJacSolve()`, and restore `alpha` afterwards. That drops the
   `X` mass term, leaving `dF/dy`.

3. **Substitute the identity.** Per cell, zero the `u` row of `MX` and put the
   identity in its `u` column. **Mirror the block indices from
   `updateMatricesForJacSolve`; do not retype them from this plan** — the layout
   is `[sigma | q | u | aux]` and the `u` row starts at `2 * nVars * (k+1)`.

4. **Build the RHS and solve.** Per the spec's four equations: the `sigma` row
   gets `-Pi(dSigmaHat/du u' + dSigmaHat/dt)`, the `q` row `B^T u' + dRF/dt`, the
   `aux` row `-Pi(dG/du u' + dG/dt)`, the `u` row the known `u'` itself, and the
   `lambda` row `-G_c u' + dL/dt`. Then `solveJacEq(rhs, out)` and copy `out`
   into `dydtComplete`.

- [ ] **Step 4: Run the tests**

```sh
make test -j$(nproc) 2>&1 | tail -6
```

Expected: both new cases pass. If `the_u_block_round_trips_through_the_identity_row`
fails, the identity substitution or the condensation is wrong — fix that before
looking at the algebraic blocks, whose values depend on it.

- [ ] **Step 5: Check the factorisation risk**

```sh
make regression_tests 2>&1 | tail -4
```

Expected: bit-identical. This is where the spec's main risk shows up:
`updateMatricesForJacSolve` factorises `MX` in place, so assembling at
`alpha = 0` destroys the forward factors. It runs before IDA's first Newton
solve and IDA calls `JacSetup` again, so it should be harmless — if regression
moves, give the `alpha = 0` assembly its own storage rather than reusing `MX`.

- [ ] **Step 6: Commit**

```bash
git add AlgebraicDerivatives.cpp SystemSolver.hpp Makefile Tests/UnitTests
git commit -m "Solve the differentiated constraints for q', sigma' and phi'"
```

---

### Task 3: Point the gate at it, and check it against the reverted branch

**Files:** Modify `Solver.cpp` (`initialize()`, `objectiveIsDecreasing()`); Test `Tests/UnitTests/AlgebraicDerivativeTests.cpp`

**Interfaces:**
- Consumes: `computeAlgebraicTimeDerivatives()` from Task 2.
- Produces: a gate whose dG/dt uses all four terms.

- [ ] **Step 1: Write the two failing tests**

The first reuses `QIntegralObjective` from the reverted branch — recover it with
`git show dgdt-gate-after-step:Tests/UnitTests/SolverLifecycleTests.cpp` and copy
the class across:

```cpp
BOOST_AUTO_TEST_CASE(a_q_only_objective_has_a_nonzero_dGdt_at_t0)
{
    // The whole point. g depends on q alone, so dG/dt = Int q' dx, which is
    // exactly zero when the gate reads IDA's dydt at t0 -- such a run could
    // never be rejected however badly its objective was falling.
    Grid grid(0.0, 1.0, nCells);
    TestDiffusion problem(alg_config);
    QIntegralObjective objective(1.0);
    SystemSolver sys(grid, k, &problem);
    /* ...configure as above, stem "algderiv_qgate"... */
    sys.setAdjointProblem(&objective);
    sys.setObjectiveDecreaseTolerance(1e-12);

    {
        CapturedOutput quiet;
        sys.initialize();
    }

    BOOST_TEST(sys.objectiveIsDecreasing() || !sys.objectiveIsDecreasing());  // force evaluation
    BOOST_TEST(sys.lastDGdt()(0) != 0.0,
               "dG/dt for a q-only objective is still exactly zero, so the gate is "
               "reading IDA's dydt rather than the completed one");

    {
        CapturedOutput quiet;
        sys.destroySundials();
    }
}
```

The second is the cross-check against the reverted approach, and is the only
test that compares this computation against an independent one:

```cpp
BOOST_AUTO_TEST_CASE(it_agrees_with_the_derivative_one_ida_step_in)
{
    // Two entirely different routes to the same quantity: this one
    // differentiates the constraints at t0, the other reads IDA's own dYdt
    // after a single step. They must agree as the step shrinks.
    //
    // The stepped value is what branch dgdt-gate-after-step computed; this is
    // the check that made the complete fix worth preferring to it.
    //
    // Tolerance is loose because the comparison is first-order in the step IDA
    // chooses, which is not under our control -- tighten it only if the step is
    // pinned with setInitialTimestep.
    /* ... build two solvers on the same problem and objective ...
       A: initialize(); computeAlgebraicTimeDerivatives(); dGdt at t0
       B: initialize(); one IDASolve(..., IDA_ONE_STEP); dGdt at t1
       BOOST_TEST(a == b, tolerance(1e-3)); ... */
}
```

Write B's step with `setInitialTimestep` set small, so the comparison is against
a known step rather than IDA's choice.

- [ ] **Step 2: Run and watch them fail**

Expected: `a_q_only_objective_has_a_nonzero_dGdt_at_t0` fails with dG/dt exactly
zero — the gate still reads IDA's `dydt`.

- [ ] **Step 3: Wire the gate**

In `initialize()`, after seeding `dydtComplete`:

```cpp
	// Only when the gate is armed: this is an extra assembly and factorisation,
	// and nothing else reads the algebraic blocks.
	if (CheckObjectiveDecrease)
		computeAlgebraicTimeDerivatives();
```

In `objectiveIsDecreasing()`, replace `dGdt(gIndex)` — which is
`dGdt(gIndex, y, dydt)` — with `dGdt(gIndex, y, dydtComplete)`.

- [ ] **Step 4: Run everything**

```sh
export PATH="$PWD/.venv/bin:$PATH"
make test && make regression_tests && make python_tests && make typecheck && make stubs-check
```

Expected: all pass; regression bit-identical.

- [ ] **Step 5: Falsify the new tests**

Temporarily point `objectiveIsDecreasing()` back at `dGdt(gIndex)` and rerun.
`a_q_only_objective_has_a_nonzero_dGdt_at_t0` must fail with exactly zero, and
`it_agrees_with_the_derivative_one_ida_step_in` must fail too. A test that passes
either way is not evidence. Restore afterwards.

- [ ] **Step 6: Commit**

```bash
git add Solver.cpp Tests/UnitTests/AlgebraicDerivativeTests.cpp
git commit -m "Evaluate the dG/dt gate against the completed derivative"
```

---

### Task 4: The manufactured-solution check, and documentation

**Files:** Modify `Tests/UnitTests/AlgebraicDerivativeTests.cpp`, `TODO`, `docs/adjoints.rst`, `CLAUDE.md`

- [ ] **Step 1: Add the manufactured-solution test**

Every test so far checks that the derivatives are nonzero, self-consistent, or
close to an independent estimate. None checks they are *right*. Build a case with
`u(x, t)` known in closed form — `MMSConvergenceTests.cpp` has the machinery —
so that `u' `, `q' = d(u')/dx` and `sigma' = d(sigmaHat)/dt` follow analytically,
and compare each block of `dydtComplete` against them. Read
`MMSConvergenceTests.cpp`'s `ManufacturedDiffusion` and mirror its structure;
note the sign convention trap in CLAUDE.md — the stored `sigma` is `-sigmaHat`,
so the expected `sigma'` is negated too.

- [ ] **Step 2: Add the two boundary tests**

An autonomous case must difference to **exactly** zero in the explicit `d/dt`
terms, not merely small. And a case whose `LowerBoundary` varies in `t` must
produce a nonzero contribution — with `RF_cellwise` back at its `t0` value
afterwards, which is worth asserting directly since restoring it is a manual
step in the implementation.

- [ ] **Step 3: Rewrite the TODO entry**

`TODO:73-89` describes both fixes with the complete one open. Record that it is
done, that the derivatives are second-order accurate in the differencing step for
a case with explicit time dependence and exact otherwise, and that the one-step
alternative was implemented and reverted on branch `dgdt-gate-after-step`.

- [ ] **Step 4: Document in `docs/adjoints.rst` and `CLAUDE.md`**

`docs/adjoints.rst:133-158` says the gate evaluates after the initial condition
is built — still true, and now correct rather than partial. Add why: the
algebraic blocks are solved for rather than read. `CLAUDE.md` gains an entry on
`dydtComplete` versus `dYdt` and on the identity substitution, both of which look
like tidying to undo.

- [ ] **Step 5: Run everything, including the docs build**

```sh
make test && make regression_tests && make python_tests && make typecheck && make stubs-check
python3 -m venv /tmp/docsvenv-alg && /tmp/docsvenv-alg/bin/pip install -q -r docs/requirements.txt
/tmp/docsvenv-alg/bin/sphinx-build -W -j auto -b html docs docs/_build/html 2>&1 | tail -4
rm -rf docs/_build
```

- [ ] **Step 6: Commit**

```bash
git add Tests/UnitTests TODO docs CLAUDE.md
git commit -m "Check the algebraic derivatives against a manufactured solution"
```

---

## Verification summary

| Claim | How it is checked |
|---|---|
| Separate storage from IDA's | `dydtComplete_starts_as_a_copy_of_idas_derivative` |
| The identity substitution is sound | `the_u_block_round_trips_through_the_identity_row` |
| The blocks are populated | `the_algebraic_blocks_stop_being_zero` |
| The gate sees them | `a_q_only_objective_has_a_nonzero_dGdt_at_t0`, falsified in Task 3 Step 5 |
| The values agree with an independent route | `it_agrees_with_the_derivative_one_ida_step_in` |
| The values are **right** | the manufactured-solution test, Task 4 |
| Autonomous cases are exact | the zero-difference test, Task 4 |
| The forward solve is untouched | `make regression_tests` bit-identical, at every task |

## Known hazards

1. **`updateMatricesForJacSolve` factorises in place.** Assembling at
   `alpha = 0` destroys the forward factors. Task 2 Step 5 is the check; if it
   fails, the `alpha = 0` assembly needs its own storage.
2. **The `alpha` restore must survive an exception.** If the assembly throws with
   `alpha` still zero, every later Jacobian is wrong and nothing says so. Use a
   scope guard, not a trailing assignment.
3. **`updateBoundaryConditions` restore, likewise.** Three calls in a row leave
   `RF_cellwise` at `t0 - h` until the fourth puts it back.
4. **`MMSConvergenceTests.cpp`'s sign convention.** The stored `sigma` is
   `-sigmaHat`; a manufactured `sigma'` must be negated to match. CLAUDE.md notes
   that getting this backwards still converges, at the right rate, to the wrong
   function.
