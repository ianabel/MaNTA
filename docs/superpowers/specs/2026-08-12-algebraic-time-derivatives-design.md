# Algebraic time derivatives, so dG/dt is complete at t0

Date: 2026-08-12
Status: approved, ready for an implementation plan

## Why

`SystemSolver::dGdt` assembles

```
dG/dt = Int ( dg/du u' + dg/dq q' + dg/dsigma sigma' + dg/dphi phi' ) dx
```

and is verified against a finite difference of a real objective
(`dGdt_matches_a_finite_difference_of_the_objective`). But at `t0` the `q`,
`sigma` and `phi` blocks of `dydt` are **identically zero**: those fields are
algebraic, and `IDA_YA_YDP_INIT` computes algebraic *values* and differential
*derivatives*, so there is no `y'` for them to fetch. Three of the four terms
vanish, and the dG/dt early-exit gate accepts or rejects an optimisation step on
the objective's `u` dependence alone. An objective depending only on `q` gets
exactly zero.

`at_t0_only_the_differential_part_of_dydt_exists` pins that this is structural
rather than a defect in the `id` vector.

TODO records two fixes. Taking one IDA step is the cheap one; it was implemented
and reverted (branch `dgdt-gate-after-step`, kept as the record) because it
trades the question for a different one — dG/dt one step in is the derivative of
the *stepped* state, it costs a step on every armed run, and it perturbs that
run's trajectory. **This is the complete fix**: computing the algebraic time
derivatives directly, at `t0`, from the constraints that define them.

## The derivation

The algebraic residual rows (`SystemSolver.cpp:1169-1185`, plus the `lambda` row
at `:1122`) are

```
res.sigma  = A sigma + Pi( sigmaHat(u, q, x, t) )                 = 0
res.q      = -A q - B^T u + C^T lambda - RF(t)                    = 0
res.Aux    = Pi( G(phi, u, q, sigma, x) )                         = 0
res.lambda = Csigma sigma + Cq q + G_c u + H lambda - L(t)        = 0
```

Only `res.u` carries a time derivative, through `X u'`. Differentiating the four
algebraic rows with respect to `t` and collecting the unknowns
`(q', sigma', phi', lambda')` with `u'` — which IDA has, because `u` is
differential — as data:

```
-A q'                        + C^T lambda'   =  B^T u' + dRF/dt
 A sigma' + Pi(dsigmaHat/dq q')              = -Pi( dsigmaHat/du u' + dsigmaHat/dt )
 Pi( dG/dq q' + dG/dsigma sigma' + dG/dphi phi' )
                                             = -Pi( dG/du u' + dG/dt )
 Csigma sigma' + Cq q' + H lambda'           = -G_c u' + dL/dt
```

Every block on the left is one `Matrices.cpp` already assembles. This is a
linear system in the same unknowns and with the same sparsity as the forward
Jacobian at `alpha = 0`.

## Decisions taken

1. **Substitute the `u` row with an identity and reuse `solveJacEq`.**
2. **Obtain the explicit `d/dt` terms by central difference**, with the state
   held fixed.

## 1. The linear solve

The system above omits the `u` *row* — differentiating `res.u` would bring in
`u''` — while keeping the `u` *column* as data. So it is a sub-block of `MX`,
not `MX`. Rather than write a second condensation path, the system is made
square in the shape `solveHDGJac` already handles:

* assemble the blocks with `alpha = 0`, so the `X` mass term drops out and what
  is left is `dF/dy`;
* per cell, overwrite the `u` row of `MX` with the identity;
* put the known `u'` into the `u` row of the right-hand side.

`solveHDGJac`'s static condensation onto `lambda` and its back-substitution are
then reused verbatim, and `solveJacEq`'s Woodbury border still handles the
global scalars. The `u` row solves trivially and returns `u'` unchanged, which
is also a free self-check: if the returned `u` block differs from the `u'` that
went in, the substitution or the condensation is wrong.

**This is not the forward solve's factorisation.** TODO's "one linear solve
against the same Jacobian the forward solve already builds" means the same
*blocks*, not the same *factors* — the forward factorisation is at `alpha = cj`,
and this one is at `alpha = 0` with a modified row. It needs its own assembly
and factorisation pass.

## 2. The explicit d/dt terms

`RF_cellwise` and `L_global` hold time-dependent boundary data and are filled by
`updateBoundaryConditions(t)`; `sigmaHat` and `G` take `t` directly. None of
their time derivatives is available: `TransportSystem::LowerBoundary(Index, Time)`
has no derivative counterpart, and there is no `dSigmaFn_dt` or `dAuxG_dt`.

They are obtained by central difference with the state held fixed:

```
dRF/dt ~ [ RF(t + h) - RF(t - h) ] / 2h
```

and likewise for `L`, `sigmaHat` and `G`. No physics case has to implement
anything, and an autonomous case — which is most of them — differences to zero
to round-off rather than to noise.

`h` is `sqrt(eps) * max(1, |t|)`, the standard central-difference choice, which
makes the truncation and round-off contributions comparable.

**`updateBoundaryConditions(t)` must be called again with the original `t`
afterwards.** It writes `RF_cellwise` and `L_global` in place, and those are what
the forward residual reads; leaving them at `t + h` would corrupt the run. This
is the same trap CLAUDE.md already records about those two arrays.

**The accuracy claim is bounded, not exact.** dG/dt becomes second-order accurate
in `h` rather than exact, and only in the terms that come from explicit time
dependence. For an autonomous case it is exact.

## 3. Where it runs, and what it must not touch

A new `SystemSolver::computeAlgebraicTimeDerivatives()`, called from
`initialize()` after `IDACalcIC`, writing into **a separate vector, not IDA's
`dYdt`**.

That separation is the load-bearing part. `dYdt` is the N_Vector IDA takes its
first step from; overwriting its algebraic entries after `IDA_YA_YDP_INIT` would
change the state the integration starts from, and the failure would appear as a
step-size or convergence problem somewhere later. The solver gains
`dydtComplete` — owning storage plus a `DGSoln` view, alongside the existing
`yJac`/`dydtJac` — holding `u'` copied from `dYdt` and the algebraic blocks
computed here.

`objectiveIsDecreasing()` then evaluates `dGdt(gIndex, y, dydtComplete)` instead
of `dGdt(gIndex)`. Nothing else reads it. The forward solve is untouched by
construction, which is what keeps the regression suite bit-identical.

The gate itself stays exactly where it is, between `initialize()` and
`integrate()` — no step, no trajectory perturbation, no cadence question. That
was the whole point of preferring this fix.

## 4. Testing

**The reference that makes this checkable is the reverted branch.** The
one-step gate computes dG/dt at `t1` from a fully populated `dYdt`. As the step
shrinks, that value approaches the one computed here at `t0`. A test that runs
both and requires them to agree to the step size is the strongest available
check, because the two are computed by entirely different routes — one from
IDA's own derivative after a step, one from differentiating the constraints.

Around it:

* **A q-only objective gives a nonzero dG/dt at `t0`.** This is the direct
  statement of the change: today it is exactly zero. The `QIntegralObjective`
  written for the reverted branch is reused.
* **The `u` block round-trips.** The identity substitution must return the `u'`
  that went in, bit for bit. Cheap, and it catches a condensation error that
  would otherwise show up only as a wrong gradient.
* **A manufactured solution.** For `u(x,t)` known in closed form, `q' = d(u')/dx`
  and `sigma'` follow analytically; `MMSConvergenceTests.cpp` already has the
  machinery to build such a case. This is the only test that checks the computed
  derivatives are *right* rather than merely nonzero or self-consistent.
* **An autonomous case differences to zero.** The explicit `d/dt` terms must be
  zero to round-off when the boundaries and hooks do not depend on `t`, not
  merely small.
* **A time-dependent boundary is picked up.** The converse: a case whose
  `LowerBoundary` varies in `t` must produce a nonzero `dRF/dt` contribution,
  and `RF_cellwise` must be back at its `t` value afterwards.

The regression suite must be bit-identical: nothing in the forward path changes,
and no config arms the gate.

## 5. Cost

One extra assembly and factorisation of the per-cell blocks, once per
`initialize()`, plus four cheap re-evaluations of the boundary data for the
differences. It is paid on every run that arms the gate, and on no other —
`computeAlgebraicTimeDerivatives` is called only when `CheckObjectiveDecrease`
is set.

## Out of scope

* Making the *stored* `dYdt` complete. IDA owns it and does not want it changed;
  this design deliberately computes into separate storage.
* New physics hooks for explicit time derivatives. The central difference is
  what avoids them; if a case ever needs exactness there, the hooks are the
  upgrade path and the differencing is where they would slot in.
* The gate's position. It stays between `initialize()` and `integrate()`.

## Risks

* **Re-factorising the blocks may disturb the forward solve.**
  `updateMatricesForJacSolve` factorises `MX` in place. This computation
  re-assembles at `alpha = 0` and factorises, destroying whatever factors were
  there. It runs once, at the end of `initialize()`, before IDA has taken a
  step — and IDA calls `JacSetup` before its first Newton solve — so it should
  be harmless. "Should be" is doing work in that sentence: the failure mode is
  slow Newton convergence rather than a wrong answer, which is exactly the class
  of defect this codebase has been bitten by before. The bit-identical
  regression requirement is what would catch it, and the `alpha = 0` assembly
  should use its own storage if it does not.
* **The identity substitution is a trick.** A reader who sees `MX` with an
  identity row will reasonably wonder whether the matrix is still the Jacobian.
  It is not, and the comment must say so.
* **Second-order accuracy, not exactness**, for any case with explicit time
  dependence. Stated in the docs rather than hidden; the manufactured-solution
  test is what bounds it.
